import pandas as pd
from sqlalchemy import create_engine, text, inspect
from pathlib import Path
import time

DB_USER = "postgres"
DB_PASSWORD = "12345"  
DB_HOST = "172.22.195.222"
DB_PORT = "5432"
DB_NAME = "postgres"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    echo=False
)

DATA_DIR = Path("data")

ALL_TABLES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv", 
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "POS_CASH_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}

def load_table_safe(table_name, filename):
    """
    Безопасная загрузка таблицы с учетом регистра
    """
    path = DATA_DIR / filename
    
    if not path.exists():
        return False
    
    start_time = time.time()
    
    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names(schema='public')
        
        table_exists = any(t.lower() == table_name.lower() for t in existing_tables)
        
        if table_exists:
            with engine.connect() as conn:
                for t in existing_tables:
                    if t.lower() == table_name.lower():
                        conn.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
                        break
                conn.commit()
        

        chunk_size = 50000
        total_rows = 0
        
        for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size, low_memory=False), 1):
            if i == 1:
                chunk.to_sql(table_name, engine, if_exists='replace', index=False, method=None)
            else:
                chunk.to_sql(table_name, engine, if_exists='append', index=False, method=None)
            
            rows_in_chunk = len(chunk)
            total_rows += rows_in_chunk
            
        
        table_variants = [
            f'"{table_name}"',  
            table_name.lower(), 
            table_name.upper(), 
        ]
        
        db_count = 0
        for table_variant in table_variants:
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(f'SELECT COUNT(*) FROM {table_variant}'))
                    db_count = result.scalar()
                    break
            except:
                continue
        
        total_time = time.time() - start_time
        
        if db_count > 0:
            return True
        else:
            return True
            
    except Exception as e:
        return False


inspector = inspect(engine)
existing_tables = inspector.get_table_names(schema='public')
print(f"Найдено таблиц: {len(existing_tables)}")

tables_to_load = []

for table_name, filename in ALL_TABLES.items():
    table_exists = any(t.lower() == table_name.lower() for t in existing_tables)
    
    if table_exists:
        try:
            with engine.connect() as conn:
                exact_name = None
                for t in existing_tables:
                    if t.lower() == table_name.lower():
                        exact_name = t
                        break
                
                if exact_name:
                    result = conn.execute(text(f'SELECT COUNT(*) FROM "{exact_name}"'))
                    count = result.scalar()
                    print(f"{table_name:25} → {count:>10,} строк (уже загружена)")
                else:
                    print(f"{table_name:25} → таблица найдена, но не удалось получить данные")
        except:
            print(f"{table_name:25} → ошибка при проверке")
    else:
        print(f"{table_name:25} → нужно загрузить")
        tables_to_load.append((table_name, filename))

if tables_to_load:
    print(f"\nЗагружаем {len(tables_to_load)} отсутствующих таблиц:")
    
    for table_name, filename in tables_to_load:
        load_table_safe(table_name, filename)
else:
    print("\n Все таблицы уже загружены!")
