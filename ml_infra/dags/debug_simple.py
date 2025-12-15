"""
debug_simple.py
Самый простой DAG для проверки работоспособности Airflow
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def print_hello():
    print("=" * 60)
    print("ПРОВЕРКА: Python работает в Airflow")
    print("=" * 60)
    print(f"Время: {datetime.now()}")
    print(f"Python версия: import sys; print(sys.version)")
    print("=" * 60)
    return "SUCCESS"

with DAG(
    dag_id='debug_simple',
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['debug'],
) as dag:
    
    test_task = PythonOperator(
        task_id='test_simple_print',
        python_callable=print_hello,
    )