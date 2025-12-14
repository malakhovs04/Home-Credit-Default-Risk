import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text
import sys

# Добавляем путь для импорта модулей
sys.path.append('/app')

# Получаем параметры из переменных окружения
DB_HOST = os.getenv("DB_HOST", "my-postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "12345")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
TEST_MODE = os.getenv("TEST_MODE", "true").lower() == "true"
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "3000"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "300"))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def main():
    log.info("🚀 Запуск обработки данных Home Credit")
    log.info(f"📊 Режим: {'ТЕСТОВЫЙ' if TEST_MODE else 'ПРОДУКЦИОННЫЙ'}")
    
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
        raise
    
    # Список таблиц для загрузки
    tables = [
        "application_train", "bureau", "bureau_balance",
        "previous_application", "POS_CASH_balance", 
        "installments_payments", "credit_card_balance"
    ]
    
    data = {}
    
    # Загружаем таблицы
    for table in tables:
        try:
            limit = f" LIMIT {ROW_LIMIT}" if TEST_MODE else ""
            query = f"SELECT * FROM {table}{limit}"
            
            log.info(f"📥 Загрузка таблицы: {table}")
            df = pd.read_sql(query, engine)
            df.columns = df.columns.str.lower()  # Приводим имена колонок к нижнему регистру
            data[table] = df
            
            log.info(f"✅ {table}: {df.shape[0]} строк, {df.shape[1]} колонок")
            
        except Exception as e:
            log.warning(f"⚠️ Таблица {table} не найдена: {e}")
            data[table] = pd.DataFrame()  # Пустой DataFrame
    
    # Проверяем, что основная таблица загружена
    if data.get("application_train", pd.DataFrame()).empty:
        raise Exception("❌ Основная таблица application_train не загружена!")
    
    try:
        from home_credit_features.engineer import HomeCreditFeatureEngineer
        log.info("✅ Модуль feature engineering загружен")
        
        # Создаем инженер фич
        engineer = HomeCreditFeatureEngineer(
            verbose=True, 
            sample_size=SAMPLE_SIZE if TEST_MODE else None
        )
        
        # Загружаем данные
        engineer.load_data(data)
        
        # Запускаем pipeline
        log.info("🔧 Запуск feature engineering pipeline...")
        result_df = engineer.run_full_pipeline()
        log.info(f"✅ Feature engineering завершен: {result_df.shape}")
        
        # Сохраняем результат
        output_table = "test_train_data_cleaned" if TEST_MODE else "train_data_cleaned"
        result_df.to_sql(
            output_table, 
            engine, 
            if_exists="replace", 
            index=False, 
            chunksize=5000, 
            method="multi"
        )
        
        log.info(f"🎉 УСПЕХ! Сохранено {len(result_df):,} строк → {output_table}")
        
    except ImportError as e:
        log.warning(f"⚠️ Модуль feature engineering не найден: {e}")
        log.info("ℹ️  Пропускаем feature engineering, только загрузка данных")
    except Exception as e:
        log.error(f"❌ Ошибка feature engineering: {e}")
        raise
    
    log.info("✅ Обработка завершена успешно!")

if __name__ == "__main__":
    main()