# dags/home_credit_full_preparation.py - исправленное имя таблицы POS_CASH_balance + обработка типов перед возвратом DF
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.exceptions import AirflowException
import pandas as pd
import logging
import gc
import numpy as np
import sys

sys.path.append('/opt/airflow/dags')

from home_credit_features.engineer import HomeCreditFeatureEngineer

def setup_logging():
    logger = logging.getLogger("airflow.task")
    logger.setLevel(logging.INFO)
    return logger

# ЗАДАЧА 1: ЗАГРУЗКА ДАННЫХ
def load_all_tables(**context):
    logger = setup_logging()
    logger.info("Начало загрузки таблиц из PostgreSQL")

    hook = PostgresHook(postgres_conn_id="home_credit_db")
    test_mode = context["dag_run"].conf.get("test_mode", True) if context["dag_run"].conf else True
    row_limit = 10 if test_mode else None

    data = {}

    try:
        sql = 'SELECT * FROM "application_train"'
        if row_limit:
            sql += f" ORDER BY RANDOM() LIMIT {row_limit}"
        
        app_df = hook.get_pandas_df(sql)
        
        if app_df.empty:
            raise ValueError("application_train вернула пустой результат")
        
        app_df.columns = app_df.columns.str.lower()
        sk_ids = tuple(app_df["sk_id_curr"].astype(int).tolist())
        
        data["application_train"] = app_df
        logger.info(f"Успешно загружено application_train: {app_df.shape}, IDs: {len(sk_ids)}")
        
    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке application_train: {e}")
        raise AirflowException("Не удалось загрузить основную таблицу application_train!") from e

    aux_tables = [
        ("bureau",               "bureau",               "SK_ID_CURR"),
        ("previous_application",  "previous_application",  "SK_ID_CURR"),
        ("credit_card_balance",   "credit_card_balance",   "SK_ID_CURR"),
        ("installments_payments", "installments_payments", "SK_ID_CURR"),
        ("pos_cash_balance",      "POS_CASH_balance",      "SK_ID_CURR"),   
    ]

    for display_name, table_name, id_col in aux_tables:
        try:
            sql = f'''
                SELECT * FROM "{table_name}"
                WHERE "{id_col}" = ANY(ARRAY[{",".join(map(str, sk_ids))}])
            '''
            df = hook.get_pandas_df(sql)
            df.columns = df.columns.str.lower()
            data[display_name] = df
            logger.info(f"Загружено {display_name}: {df.shape}")
            del df
            gc.collect()
        except Exception as e:
            logger.warning(f"Не удалось загрузить {display_name} (таблица может отсутствовать или быть пустой): {e}")
            data[display_name] = pd.DataFrame()

    if "bureau" in data and not data["bureau"].empty:
        bureau_ids = tuple(data["bureau"]["sk_id_bureau"].astype(int).tolist())
        try:
            sql = f'''
                SELECT * FROM "bureau_balance"
                WHERE "SK_ID_BUREAU" = ANY(ARRAY[{",".join(map(str, bureau_ids))}])
            '''
            df = hook.get_pandas_df(sql)
            df.columns = df.columns.str.lower()
            data["bureau_balance"] = df
            logger.info(f"Загружено bureau_balance: {df.shape}")
            del df
            gc.collect()
        except Exception as e:
            logger.warning(f"Не удалось загрузить bureau_balance: {e}")
            data["bureau_balance"] = pd.DataFrame()
    else:
        data["bureau_balance"] = pd.DataFrame()
        logger.info("bureau пустая → bureau_balance пропущена")

    logger.info("Загрузка всех таблиц завершена")
    for name, df in data.items():
        logger.info(f" → {name}: {df.shape if not df.empty else 'пусто'}")

    return data

#ЗАДАЧА 2: ПОЛНЫЙ ПАЙПЛАН
def process_and_engineer(**context):
    logger = setup_logging()
    data_dict = context["ti"].xcom_pull(task_ids="load_data")

    if not data_dict or "application_train" not in data_dict or data_dict["application_train"].empty:
        raise AirflowException("Основная таблица application_train не загружена!")

    test_mode = context["dag_run"].conf.get("test_mode", True) if context["dag_run"].conf else True
    sample_size = 500 if test_mode else None

    engineer = HomeCreditFeatureEngineer(verbose=True, sample_size=sample_size)
    engineer.load_data(data_dict)

    final_df = engineer.run_full_pipeline()
    
    for col in final_df.select_dtypes(include=['object']).columns:
        final_df[col] = final_df[col].astype('str').fillna('')

    logger.info(f"Пайплайн завершён! Итоговый датафрейм: {final_df.shape}")
    return final_df

#ЗАДАЧА 3: СОХРАНЕНИЕ
def save_to_postgres(**context):
    logger = setup_logging()
    final_df: pd.DataFrame = context["ti"].xcom_pull(task_ids="process_data")

    test_mode = context["dag_run"].conf.get("test_mode", True) if context["dag_run"].conf else True
    table_name = "test_train_data_cleaned" if test_mode else "train_data_cleaned"

    hook = PostgresHook(postgres_conn_id="home_credit_db")
    engine = hook.get_sqlalchemy_engine()

    with engine.begin() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")

    chunksize = 5000
    final_df.to_sql(
        name=table_name,
        con=engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
        method="multi"
    )

    row_count = hook.get_first(f"SELECT COUNT(*) FROM {table_name}")[0]
    logger.info(f"Данные успешно сохранены в таблицу {table_name} — {row_count:,} строк")

# DAG
default_args = {
    "owner": "you",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2023, 1, 1),
}

with DAG(
    dag_id="home_credit_full_preparation",
    default_args=default_args,
    description="Полная подготовка данных Home Credit → train_data_cleaned",
    schedule_interval=None,
    catchup=False,
    tags=["home_credit", "feature_engineering"],
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=4),
) as dag:

    t1 = PythonOperator(
        task_id="load_data",
        python_callable=load_all_tables,
        provide_context=True,
    )

    t2 = PythonOperator(
        task_id="process_data",
        python_callable=process_and_engineer,
        provide_context=True,
    )

    t3 = PythonOperator(
        task_id="save_data",
        python_callable=save_to_postgres,
        provide_context=True,
    )

    t1 >> t2 >> t3
