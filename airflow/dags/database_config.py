from airflow import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator
import churn_utils

with DAG(
    dag_id='config_db_v9',
    description='',
    start_date=datetime(2024, 5, 28),
    schedule_interval='@once'
) as dag:
    task1 = PythonOperator(
        task_id='conf',
        python_callable=churn_utils.conf_db
    )
