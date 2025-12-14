from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

with DAG(
    dag_id="docker_home_credit_preparation",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["home-credit", "docker", "data-processing"],
    max_active_runs=1,
) as dag:

    process_data = DockerOperator(
        task_id="process_home_credit_data",
        image="home-credit-processor:latest",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="home-credit-network",  # Сеть из docker-compose
        mount_tmp_dir=False,
        environment={
            # Режим работы
            "TEST_MODE": "true",  # "false" для полной обработки
            "ROW_LIMIT": "100",   # Сколько строк брать из каждой таблицы (в тестовом режиме)
            "SAMPLE_SIZE": "500", # Размер выборки для feature engineering
            
            # Подключение к вашей базе данных
            "DB_HOST": "my-postgres",
            "DB_USER": "postgres",
            "DB_PASSWORD": "12345",
            "DB_NAME": "postgres",
            "DB_PORT": "5432",
            
            # Для отладки
            "PYTHONUNBUFFERED": "1",
        },
    )

    process_data