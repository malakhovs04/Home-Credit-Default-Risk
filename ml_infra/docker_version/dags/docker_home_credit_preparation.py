
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.exceptions import AirflowException

import pandas as pd
import logging
import gc

from home_credit_features.engineer import HomeCreditFeatureEngineer


def load_half_data(**context):
    """Загрузка примерно половины данных (50% от application_train + все связанные записи)"""
    logger = logging.getLogger("airflow.task")
    logger.setLevel(logging.INFO)

    hook = PostgresHook(postgres_conn_id="home_credit_db")

    data_dict = {}

    try:
        # Загружаем ВСЕ строки из application_train, но потом берём только половину
        logger.info("Загружаем все строки из application_train для выборки половины...")
        full_app_df = hook.get_pandas_df('SELECT * FROM "application_train"')
        full_app_df.columns = full_app_df.columns.str.lower()

        total_rows = len(full_app_df)
        half_rows = total_rows // 10 

        logger.info(f"Всего строк в application_train: {total_rows:,}")
        logger.info(f"Берём случайную половину: {half_rows:,} строк")

        # Случайная выборка половины строк
        app_df = full_app_df.sample(n=half_rows, random_state=42).reset_index(drop=True)

        sk_ids = tuple(app_df["sk_id_curr"].astype(int).tolist())
        data_dict["application_train"] = app_df
        logger.info(f"Выбрано application_train: {app_df.shape[0]:,} строк")

        del full_app_df
        gc.collect()

        # Вспомогательные таблицы — только для выбранных sk_id_curr
        aux_tables = [
            ("bureau", "bureau", "SK_ID_CURR"),
            ("previous_application", "previous_application", "SK_ID_CURR"),
            ("pos_cash_balance", "pos_cash_balance", "SK_ID_CURR"),
            ("installments_payments", "installments_payments", "SK_ID_CURR"),
            ("credit_card_balance", "credit_card_balance", "SK_ID_CURR"),
        ]

        for name, table, id_col in aux_tables:
            try:
                if not sk_ids:
                    data_dict[name] = pd.DataFrame()
                    continue

                placeholders = ",".join(map(str, sk_ids))
                sql = f'SELECT * FROM "{table}" WHERE "{id_col}" = ANY(ARRAY[{placeholders}])'
                df = hook.get_pandas_df(sql)
                df.columns = df.columns.str.lower()
                data_dict[name] = df
                logger.info(f"Загружено {name}: {df.shape[0]:,} строк")
                del df
                gc.collect()
            except Exception as e:
                logger.warning(f"Не удалось загрузить {name}: {e}")
                data_dict[name] = pd.DataFrame()

        # bureau_balance
        if "bureau" in data_dict and not data_dict["bureau"].empty:
            bureau_ids = tuple(data_dict["bureau"]["sk_id_bureau"].astype(int).tolist())
            try:
                placeholders = ",".join(map(str, bureau_ids))
                sql = f'SELECT * FROM "bureau_balance" WHERE "SK_ID_BUREAU" = ANY(ARRAY[{placeholders}])'
                df = hook.get_pandas_df(sql)
                df.columns = df.columns.str.lower()
                data_dict["bureau_balance"] = df
                logger.info(f"Загружено bureau_balance: {df.shape[0]:,} строк")
                del df
                gc.collect()
            except Exception as e:
                logger.warning(f"Не удалось загрузить bureau_balance: {e}")
                data_dict["bureau_balance"] = pd.DataFrame()
        else:
            data_dict["bureau_balance"] = pd.DataFrame()

        logger.info("Половина данных успешно загружена")
        return data_dict

    except Exception as e:
        logger.error(f"Ошибка при загрузке половины данных: {e}", exc_info=True)
        raise AirflowException("Не удалось загрузить данные") from e


def process_half_data(**context):
    """Полный пайплайн на половине данных"""
    logger = logging.getLogger("airflow.task")
    logger.setLevel(logging.INFO)

    data_dict = context["ti"].xcom_pull(task_ids="load_half_data")

    if not data_dict or "application_train" not in data_dict or data_dict["application_train"].empty:
        raise AirflowException("Нет данных для обработки!")

    logger.info("Запуск полного пайплайна на ~50% данных")

    try:
        engineer = HomeCreditFeatureEngineer(verbose=True, sample_size=None)  # обрабатываем всё из загруженного
        engineer.load_data(data_dict)
        final_df = engineer.run_full_pipeline()

        logger.info(f"Пайплайн завершён! Итоговый размер: {final_df.shape[0]:,} строк × {final_df.shape[1]:,} колонок")

        # Приведение object-колонок
        for col in final_df.select_dtypes(include=["object"]).columns:
            final_df[col] = final_df[col].astype("str").fillna("")

        # Сохранение
        output_table = "train_data_half_cleaned"

        hook = PostgresHook(postgres_conn_id="home_credit_db")
        engine = hook.get_sqlalchemy_engine()

        with engine.begin() as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{output_table}" CASCADE')

        final_df.to_sql(
            name=output_table,
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=10000,
        )

        saved_rows = hook.get_first(f'SELECT COUNT(*) FROM "{output_table}"')[0]
        logger.info(f"Успешно сохранено {saved_rows:,} строк в таблицу '{output_table}'")

        return {
            "status": "success",
            "original_rows": len(data_dict["application_train"]),
            "final_rows": final_df.shape[0],
            "features": final_df.shape[1],
            "table": output_table,
        }

    except Exception as e:
        logger.error(f"Ошибка в пайплайне: {e}", exc_info=True)
        raise AirflowException(f"Пайплайн упал: {str(e)}") from e


# ==============================
# DAG
# ==============================

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="home_credit_half_preparation",
    default_args=default_args,
    description="Подготовка ~50% данных Home Credit (без OOM)",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["home_credit", "half", "feature_engineering"],
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=6),
) as dag:

    load_task = PythonOperator(
        task_id="load_half_data",
        python_callable=load_half_data,
        provide_context=True,
    )

    process_task = PythonOperator(
        task_id="process_half_data",
        python_callable=process_half_data,
        provide_context=True,
        execution_timeout=timedelta(hours=4),
    )

    load_task >> process_task