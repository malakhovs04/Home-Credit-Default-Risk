# train_model.py - БЕЗ database_connector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sys
import os
import warnings
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')

# Добавляем пути для импорта
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/home_credit_features')

print(f"Python path: {sys.path}")

# Пробуем импортировать только model_trainer_mlflow
try:
    from model_trainer_mlflow import ModelTrainerMLflow
    print("✅ ModelTrainerMLflow успешно импортирован")
except ImportError as e:
    print(f"❌ Ошибка импорта ModelTrainerMLflow: {e}")
    sys.exit(1)

def load_data_from_postgres(table_name="test_train_data_cleaned"):
    """Загружает данные напрямую из PostgreSQL без database_connector"""
    print(f"📥 Загрузка данных из таблицы: {table_name}")
    
    # Параметры подключения из переменных окружения
    db_host = os.getenv('DB_HOST', 'my-postgres')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', '12345')
    db_name = os.getenv('DB_NAME', 'postgres')
    db_port = os.getenv('DB_PORT', '5432')
    
    # Создаем строку подключения
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    try:
        # Создаем engine
        engine = create_engine(connection_string)
        
        # Проверяем подключение
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"✅ Подключение к БД успешно: {db_host}:{db_port}/{db_name}")
        
        # Загружаем данные
        row_limit = os.getenv('ROW_LIMIT', '100')
        query = f"SELECT * FROM {table_name} LIMIT {row_limit}"
        print(f"📊 Выполняем запрос: {query}")
        
        df = pd.read_sql(query, engine)
        print(f"✅ Загружено {df.shape[0]} строк, {df.shape[1]} колонок")
        
        return df
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        
        # Если не удалось подключиться, используем тестовые данные
        print("🔄 Используем тестовые данные...")
        return generate_test_data()

def generate_test_data():
    """Генерирует тестовые данные, если нет доступа к БД"""
    n_samples = 100
    n_features = 95
    
    # Генерируем случайные данные
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Добавляем целевую переменную
    data['target'] = np.random.randint(0, 2, n_samples)
    
    # Добавляем несколько категориальных признаков для теста
    data['category_1'] = np.random.choice(['A', 'B', 'C'], n_samples)
    data['category_2'] = np.random.choice(['X', 'Y'], n_samples)
    
    print(f"✅ Сгенерировано тестовых данных: {data.shape}")
    return data

def apply_one_hot_encoding(X_train, X_val):
    """
    Применяет One-Hot Encoding к категориальным признакам
    """
    print("🧹 Применяем One-Hot Encoding к категориальным признакам...")
    
    # Находим категориальные признаки
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        print("⚠️ Категориальных признаков не найдено")
        return X_train, X_val, X_train.columns.tolist()
    
    print(f"🔍 Найдено категориальных признаков: {len(categorical_cols)}")
    
    # Создаем OneHotEncoder
    ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore',
        drop='first'
    )
    
    # Кодируем
    X_train_cat_encoded = ohe.fit_transform(X_train[categorical_cols])
    X_val_cat_encoded = ohe.transform(X_val[categorical_cols])
    
    # Получаем имена новых признаков
    new_categorical_columns = ohe.get_feature_names_out(categorical_cols)
    
    # Удаляем исходные категориальные колонки
    X_train_numeric = X_train.drop(columns=categorical_cols)
    X_val_numeric = X_val.drop(columns=categorical_cols)
    
    # Создаем DataFrame для закодированных признаков
    X_train_cat_df = pd.DataFrame(
        X_train_cat_encoded, 
        columns=new_categorical_columns, 
        index=X_train_numeric.index
    )
    
    X_val_cat_df = pd.DataFrame(
        X_val_cat_encoded, 
        columns=new_categorical_columns, 
        index=X_val_numeric.index
    )
    
    # Объединяем
    X_train_encoded = pd.concat([X_train_numeric, X_train_cat_df], axis=1)
    X_val_encoded = pd.concat([X_val_numeric, X_val_cat_df], axis=1)
    
    print(f"✅ One-Hot Encoding применен. Итоговые признаки: {X_train_encoded.shape[1]}")
    
    return X_train_encoded, X_val_encoded, X_train_encoded.columns.tolist()

def main():
    print("🚀 ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ С MLflow")
    print("=" * 60)
    
    # Параметры запуска
    model_type = 'catboost'  # Всегда используем CatBoost
    table_name = os.getenv("TABLE_NAME", "test_train_data_cleaned")
    experiment_name = os.getenv("EXPERIMENT_NAME", "home-credit-default-risk")
    
    print(f"📊 Параметры запуска:")
    print(f"  Model type: {model_type}")
    print(f"  Table name: {table_name}")
    print(f"  Experiment: {experiment_name}")
    
    # Загрузка данных (прямое подключение к PostgreSQL)
    data = load_data_from_postgres(table_name)
    
    # Подготовка данных
    target_col = "target"
    if target_col not in data.columns:
        print(f"⚠️ Целевая переменная '{target_col}' не найдена, используем первый столбец")
        target_col = data.columns[0]
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    print(f"🎯 Целевая переменная: {target_col}")
    print(f"📊 Размеры: X={X.shape}, y={y.shape}")
    
    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📈 Разделение: train={X_train.shape}, val={X_val.shape}")
    
    # Применяем One-Hot Encoding
    X_train_encoded, X_val_encoded, feature_names = apply_one_hot_encoding(X_train, X_val)
    
    # Создание модели CatBoost
    print("🤖 Создание модели CatBoostClassifier...")
    try:
        from catboost import CatBoostClassifier
        
        model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            verbose=False,
            random_seed=42,
            task_type='CPU'
        )
        print("✅ Модель CatBoost создана")
        
    except Exception as e:
        print(f"❌ Ошибка создания CatBoost: {e}")
        sys.exit(1)
    
    # Обучение с MLflow
    print("🚀 Начинаем обучение с MLflow...")
    try:
        trainer = ModelTrainerMLflow(
            model=model,
            experiment_name=experiment_name,
            run_name=f"{model_type}_training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        trainer.fit_with_mlflow(
            X_train=X_train_encoded,
            y_train=y_train,
            X_val=X_val_encoded,
            y_val=y_val,
            log_params=True,
            log_metrics=True,
            log_model=True,
            log_artifacts=True
        )
        
        print("✅ Обучение успешно завершено!")
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()