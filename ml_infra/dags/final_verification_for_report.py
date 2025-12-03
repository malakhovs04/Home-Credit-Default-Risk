from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

def format_and_print_results(**context):
    """Форматируем и выводим результаты проверки"""
    print("=" * 70)
    print("ИТОГОВАЯ ПРОВЕРКА ПОДКЛЮЧЕНИЯ AIRFLOW К POSTGRESQL")
    print("=" * 70)
    
    # Результат из первой задачи
    conn_result = context['ti'].xcom_pull(task_ids='check_connection')
    if conn_result:
        row = conn_result[0]
        print(f"   Подключение установлено!")
        print(f"   База данных: {row[1]}")
        print(f"   Пользователь: {row[2]}")
        print(f"   Сервер: {row[3]}")
        print(f"   Время: {row[4]}")
    
    # Результат из второй задачи
    tables_result = context['ti'].xcom_pull(task_ids='check_tables')
    if tables_result:
        print(f"\n📊 Найдено таблиц Home Credit: {len(tables_result)}")
        for row in tables_result:
            print(f"   - {row[0]}: {row[1]} колонок, ~{row[2]} строк")
    
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ЗАВЕРШЕНА УСПЕШНО!")
    print("Airflow корректно подключен к PostgreSQL с данными Home Credit")
    print("=" * 70)
    
    return "Отчёт сформирован"

with DAG(
    dag_id='final_connection_report',
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=['report', 'home_credit', 'postgresql'],
) as dag:
    
    # 1. Проверка подключения
    task_conn = PostgresOperator(
        task_id='check_connection',
        postgres_conn_id='home_credit_db',
        sql="""
        SELECT 
            'success' as status,
            current_database() as database,
            current_user as username,
            inet_server_addr() as server_ip,
            NOW() as check_time
        """,
        do_xcom_push=True,
    )
    
    # 2. Проверка таблиц
    task_tables = PostgresOperator(
        task_id='check_tables',
        postgres_conn_id='home_credit_db',
        sql="""
        SELECT 
            t.table_name,
            (SELECT COUNT(*) FROM information_schema.columns c WHERE c.table_name = t.table_name) as columns_count,
            (SELECT n_live_tup FROM pg_stat_user_tables s WHERE s.relname = t.table_name) as estimated_rows
        FROM information_schema.tables t
        WHERE table_schema = 'public'
        AND table_name IN ('application_train', 'application_test', 'bureau', 'previous_application')
        ORDER BY table_name
        """,
        do_xcom_push=True,
    )
    
    # 3. Формирование отчёта
    task_report = PythonOperator(
        task_id='generate_report',
        python_callable=format_and_print_results,
    )
    
    # Порядок выполнения
    task_conn >> task_tables >> task_report