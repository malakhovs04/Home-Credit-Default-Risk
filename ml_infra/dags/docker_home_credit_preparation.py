# dags/docker_home_credit_preparation.py
from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

with DAG(
    dag_id="docker_home_credit_preparation",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["home-credit", "docker"],
    max_active_runs=1,
) as dag:

    run_pipeline = DockerOperator(
        task_id="run_full_pipeline_in_docker",
        image="home-credit-processor:latest",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        environment={
            "TEST_MODE": "true",
            "ROW_LIMIT": "100",
            "SAMPLE_SIZE": "500",
            "DB_HOST": "postgres",
            "DB_USER": "airflow",
            "DB_PASSWORD": "airflow",
            "DB_NAME": "airflow",
        },
    )

    run_pipeline