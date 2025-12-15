# dags/home_credit_model_training.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.dummy import DummyOperator
import pendulum

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': pendulum.datetime(2024, 1, 1, tz="UTC"),
}

with DAG(
    dag_id='home_credit_model_training',
    default_args=default_args,
    description='DAG for training Home Credit models with OHE and MLflow',
    schedule_interval=None,
    catchup=False,
    tags=['home-credit', 'mlflow', 'ohe', 'catboost'],
    max_active_runs=1,
) as dag:
    
    start = DummyOperator(task_id='start')
    
    # Задача обучения модели с One-Hot Encoding
    train_model = DockerOperator(
        task_id='train_model',
        image='home-credit-processor:latest',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='home-credit-network',
        environment={
            # Параметры подключения к БД
            'DB_HOST': 'my-postgres',
            'DB_USER': 'postgres',
            'DB_PASSWORD': '12345',
            'DB_NAME': 'postgres',
            'DB_PORT': '5432',
            
            # Параметры обучения
            'MODEL_TYPE': 'catboost',  # Можно менять на lightgbm, xgboost
            'TABLE_NAME': 'test_train_data_cleaned',
            'EXPERIMENT_NAME': 'home-credit-default-risk-ohe',
            'ROW_LIMIT': '100',
            'TEST_MODE': 'true',
            
            # Параметры MLflow
            'LOG_TO_MLFLOW': 'true',
            'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
            'MLFLOW_DISABLE_HOST_CHECK': 'true',
            
            # Системные параметры
            'PYTHONUNBUFFERED': '1',
            'GIT_PYTHON_REFRESH': 'quiet',
        },
        mount_tmp_dir=False,
        command='python /app/train_model.py',
    )
    
    success = DummyOperator(task_id='success')
    
    start >> train_model >> success