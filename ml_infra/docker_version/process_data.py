import os
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь для импорта модулей
sys.path.append('/app')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Получаем параметры из переменных окружения
DB_HOST = os.getenv("DB_HOST", "my-postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "12345")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
TEST_MODE = os.getenv("TEST_MODE", "true").lower() == "true"
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "3000"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "300"))

# Параметры MLflow (опционально)
LOG_TO_MLFLOW = os.getenv("LOG_TO_MLFLOW", "false").lower() == "true"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

def setup_mlflow():
    """Настройка MLflow для логирования процесса обработки данных"""
    if not LOG_TO_MLFLOW:
        return None
    
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("home-credit-data-preparation")
        
        # Создаем run с информацией о запуске
        run_name = f"data_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = mlflow.start_run(run_name=run_name)
        
        # Логируем параметры запуска
        mlflow.log_params({
            "test_mode": TEST_MODE,
            "row_limit": ROW_LIMIT,
            "sample_size": SAMPLE_SIZE,
            "db_host": DB_HOST,
            "db_name": DB_NAME
        })
        
        log.info(f"✅ MLflow логирование включено. Run: {run_name}")
        return mlflow, run
    except ImportError:
        log.warning("⚠️ MLflow не установлен. Пропускаем логирование.")
        return None
    except Exception as e:
        log.error(f"❌ Ошибка инициализации MLflow: {e}")
        return None

def log_to_mlflow(mlflow_obj, metrics=None, artifacts=None):
    """Логирование метрик и артефактов в MLflow"""
    if not mlflow_obj or not LOG_TO_MLFLOW:
        return
    
    mlflow, run = mlflow_obj
    
    try:
        if metrics:
            mlflow.log_metrics(metrics)
        
        if artifacts:
            for artifact_path, local_path in artifacts.items():
                if os.path.exists(local_path):
                    mlflow.log_artifact(local_path, artifact_path)
    except Exception as e:
        log.warning(f"⚠️ Ошибка логирования в MLflow: {e}")

def end_mlflow_run(mlflow_obj, status="FINISHED"):
    """Завершение run в MLflow"""
    if not mlflow_obj:
        return
    
    mlflow, run = mlflow_obj
    try:
        mlflow.end_run(status=status)
        log.info(f"✅ MLflow run завершен со статусом: {status}")
    except Exception as e:
        log.warning(f"⚠️ Ошибка завершения MLflow run: {e}")

def load_tables(engine, tables):
    """Загрузка таблиц из базы данных"""
    data = {}
    table_stats = {}
    
    for table in tables:
        try:
            limit = f" LIMIT {ROW_LIMIT}" if TEST_MODE else ""
            query = f"SELECT * FROM {table}{limit}"
            
            log.info(f"📥 Загрузка таблицы: {table}")
            start_time = time.time()
            
            df = pd.read_sql(query, engine)
            df.columns = df.columns.str.lower()
            data[table] = df
            
            load_time = time.time() - start_time
            table_stats[table] = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "load_time_seconds": round(load_time, 2)
            }
            
            log.info(f"✅ {table}: {df.shape[0]} строк, {df.shape[1]} колонок, время: {load_time:.2f}с")
            
        except Exception as e:
            log.warning(f"⚠️ Таблица {table} не найдена: {e}")
            data[table] = pd.DataFrame()
            table_stats[table] = {"rows": 0, "columns": 0, "load_time_seconds": 0}
    
    return data, table_stats

def run_feature_engineering(data, sample_size=None):
    """Запуск feature engineering pipeline"""
    try:
        from home_credit_features.engineer import HomeCreditFeatureEngineer
        
        log.info("✅ Модуль feature engineering загружен")
        
        # Создаем инженер фич
        engineer = HomeCreditFeatureEngineer(
            verbose=True, 
            sample_size=sample_size
        )
        
        # Загружаем данные
        engineer.load_data(data)
        
        # Запускаем pipeline
        log.info("🔧 Запуск feature engineering pipeline...")
        start_time = time.time()
        result_df = engineer.run_full_pipeline()
        processing_time = time.time() - start_time
        
        log.info(f"✅ Feature engineering завершен: {result_df.shape}, время: {processing_time:.2f}с")
        
        # Проверяем целевую переменную
        target_column = None
        for col in result_df.columns:
            if 'target' in col.lower():
                target_column = col
                break
        
        if target_column:
            log.info(f"🎯 Целевая переменная найдена: {target_column}")
            # Статистика по целевой переменной
            target_stats = result_df[target_column].value_counts(normalize=True)
            log.info(f"   Распределение: {dict(target_stats.round(3))}")
        else:
            log.warning("⚠️ Целевая переменная не найдена в данных")
        
        return result_df, processing_time, target_column
        
    except ImportError as e:
        log.error(f"❌ Модуль feature engineering не найден: {e}")
        raise
    except Exception as e:
        log.error(f"❌ Ошибка feature engineering: {e}")
        raise

def save_to_database(df, engine, table_name):
    """Сохранение данных в базу данных"""
    log.info(f"💾 Сохранение данных в таблицу: {table_name}")
    start_time = time.time()
    
    # Сохраняем с оптимизацией
    df.to_sql(
        table_name, 
        engine, 
        if_exists="replace", 
        index=False, 
        chunksize=5000, 
        method="multi"
    )
    
    save_time = time.time() - start_time
    log.info(f"✅ Сохранено {len(df):,} строк → {table_name}, время: {save_time:.2f}с")
    
    return save_time

def analyze_data_quality(df, mlflow_obj):
    """Анализ качества данных и создание отчетов"""
    if df.empty:
        log.warning("⚠️ DataFrame пуст, пропускаем анализ качества")
        return
    
    try:
        # Создаем временную папку для отчетов
        import tempfile
        import json
        
        temp_dir = tempfile.mkdtemp()
        
        # 1. Основная статистика
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "missing_values_total": df.isnull().sum().sum(),
            "missing_values_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        log.info(f"📊 Статистика данных:")
        for key, value in stats.items():
            log.info(f"   {key}: {value}")
        
        # 2. Сохраняем статистику в JSON
        stats_file = os.path.join(temp_dir, "data_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 3. Создаем простой отчет о пропусках
        missing_stats = df.isnull().sum().sort_values(ascending=False)
        missing_stats = missing_stats[missing_stats > 0]
        
        if len(missing_stats) > 0:
            missing_file = os.path.join(temp_dir, "missing_values.csv")
            missing_stats.to_csv(missing_file, header=['missing_count'])
            log.info(f"📝 Найдено {len(missing_stats)} колонок с пропусками")
        
        # 4. Логируем в MLflow
        if mlflow_obj and LOG_TO_MLFLOW:
            mlflow, run = mlflow_obj
            
            # Логируем метрики
            mlflow.log_metrics({
                "data_rows": stats["total_rows"],
                "data_columns": stats["total_columns"],
                "missing_percentage": round(stats["missing_values_percentage"], 2)
            })
            
            # Логируем артефакты
            artifacts = {
                "statistics": stats_file
            }
            
            if len(missing_stats) > 0:
                artifacts["missing_analysis"] = missing_file
            
            log_to_mlflow(mlflow_obj, artifacts=artifacts)
        
        log.info("✅ Анализ качества данных завершен")
        
    except Exception as e:
        log.warning(f"⚠️ Ошибка анализа качества данных: {e}")

def main():
    """Основная функция обработки данных"""
    log.info("🚀 Запуск обработки данных Home Credit")
    log.info("=" * 60)
    log.info(f"📊 Режим: {'ТЕСТОВЫЙ' if TEST_MODE else 'ПРОДУКЦИОННЫЙ'}")
    log.info(f"🔬 MLflow логирование: {'ВКЛЮЧЕНО' if LOG_TO_MLFLOW else 'ВЫКЛЮЧЕНО'}")
    log.info(f"📈 Ограничение строк: {ROW_LIMIT if TEST_MODE else 'НЕТ'}")
    log.info(f"🎯 Размер выборки: {SAMPLE_SIZE if TEST_MODE else 'НЕТ'}")
    
    start_total_time = time.time()
    
    # Настраиваем MLflow
    mlflow_obj = setup_mlflow()
    
    # Создаем строку подключения
    conn_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    log.info(f"🔗 Подключение к: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    try:
        # Подключаемся к БД
        engine = create_engine(conn_uri)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("✅ Подключение к БД успешно")
    except Exception as e:
        log.error(f"❌ Ошибка подключения к БД: {e}")
        if mlflow_obj:
            end_mlflow_run(mlflow_obj, "FAILED")
        raise
    
    # Список таблиц для загрузки
    tables = [
        "application_train", "bureau", "bureau_balance",
        "previous_application", "POS_CASH_balance", 
        "installments_payments", "credit_card_balance"
    ]
    
    try:
        # 1. Загрузка таблиц
        data, table_stats = load_tables(engine, tables)
        
        # Логируем статистику загрузки в MLflow
        if mlflow_obj and LOG_TO_MLFLOW:
            mlflow, run = mlflow_obj
            
            # Собираем метрики по таблицам
            table_metrics = {}
            for table, stats in table_stats.items():
                table_metrics[f"{table}_rows"] = stats["rows"]
                table_metrics[f"{table}_columns"] = stats["columns"]
                table_metrics[f"{table}_load_time"] = stats["load_time_seconds"]
            
            log_to_mlflow(mlflow_obj, metrics=table_metrics)
        
        # Проверяем, что основная таблица загружена
        if data.get("application_train", pd.DataFrame()).empty:
            raise Exception("❌ Основная таблица application_train не загружена!")
        
        # 2. Feature Engineering
        result_df, processing_time, target_column = run_feature_engineering(
            data, 
            sample_size=SAMPLE_SIZE if TEST_MODE else None
        )
        
        # 3. Анализ качества данных
        analyze_data_quality(result_df, mlflow_obj)
        
        # 4. Сохранение результата
        output_table = "test_train_data_cleaned" if TEST_MODE else "train_data_cleaned"
        save_time = save_to_database(result_df, engine, output_table)
        
        # 5. Логируем финальные метрики в MLflow
        if mlflow_obj and LOG_TO_MLFLOW:
            mlflow, run = mlflow_obj
            
            final_metrics = {
                "final_rows": len(result_df),
                "final_columns": len(result_df.columns),
                "processing_time_seconds": round(processing_time, 2),
                "save_time_seconds": round(save_time, 2),
                "total_time_seconds": round(time.time() - start_total_time, 2)
            }
            
            if target_column:
                target_stats = result_df[target_column].value_counts(normalize=True)
                for class_value, percentage in target_stats.items():
                    final_metrics[f"class_{int(class_value)}_percentage"] = round(percentage * 100, 2)
            
            log_to_mlflow(mlflow_obj, metrics=final_metrics)
        
        # 6. Финальный отчет
        total_time = time.time() - start_total_time
        log.info("=" * 60)
        log.info(f"🎉 УСПЕХ! Обработка данных завершена!")
        log.info(f"📊 Результат: {len(result_df):,} строк, {len(result_df.columns)} колонок")
        log.info(f"⏱️  Общее время: {total_time:.2f} секунд")
        log.info(f"💾 Сохранено в таблицу: {output_table}")
        log.info("=" * 60)
        
        # Завершаем MLflow run
        if mlflow_obj:
            end_mlflow_run(mlflow_obj, "FINISHED")
        
    except Exception as e:
        log.error(f"❌ Критическая ошибка: {e}")
        
        # Логируем ошибку в MLflow
        if mlflow_obj and LOG_TO_MLFLOW:
            try:
                mlflow, run = mlflow_obj
                mlflow.log_param("error", str(e)[:100])  # Логируем первые 100 символов ошибки
                end_mlflow_run(mlflow_obj, "FAILED")
            except:
                pass
        
        raise

if __name__ == "__main__":
    main()