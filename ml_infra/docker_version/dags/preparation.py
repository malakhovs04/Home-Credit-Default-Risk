# dags/home_credit_model_training.py
from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.dummy import DummyOperator
import pendulum

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,  # Увеличил retries до 3
    'retry_delay': timedelta(minutes=5),
    'start_date': pendulum.datetime(2024, 1, 1, tz="UTC"),
}

with DAG(
    dag_id='home_credit_model_training',
    default_args=default_args,
    description='Training Home Credit model with CatBoost and MLflow',
    schedule_interval=None,
    catchup=False,
    tags=['home-credit', 'ml', 'catboost'],
    max_active_runs=1,
) as dag:

    start = DummyOperator(task_id='start')

    train_model = DockerOperator(
        task_id='train_model',
        image='home-credit-processor:latest',
        api_version='auto',
        auto_remove='success',
        docker_url='unix:///var/run/docker.sock',
        network_mode='bridge',
        environment={
            'DB_HOST': 'host.docker.internal',
            'DB_USER': 'postgres',
            'DB_PASSWORD': '12345',
            'DB_NAME': 'postgres',
            'DB_PORT': '5432',

            'MODEL_TYPE': 'catboost',
            'TABLE_NAME': 'train_data_half_cleaned',
            'EXPERIMENT_NAME': 'home-credit-default-risk-ohe',
            'ROW_LIMIT': '1000',
            'TEST_MODE': 'false',

            'LOG_TO_MLFLOW': 'true',
            'MLFLOW_TRACKING_URI': 'http://host.docker.internal:5050',

            'PYTHONUNBUFFERED': '1',
        },
        command='python /app/train_model.py',
        mount_tmp_dir=False,
        extra_hosts={
            'host.docker.internal': 'host-gateway',
        },
        # Добавляем dns для стабильности resolution
        dns=['8.8.8.8'],
    )

    success = DummyOperator(task_id='success')

    start >> train_model >> success