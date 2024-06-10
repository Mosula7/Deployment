import os
from airflow import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator
import churn_utils

from dotenv import load_dotenv

load_dotenv()

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")
schema_name = 'churn'

with DAG(
    dag_id='config_db_v10',
    description='',
    start_date=datetime(2024, 5, 28),
    schedule_interval='@once'
) as dag:
    task1 = PythonOperator(
        task_id='conf',
        python_callable=churn_utils.conf_db,
        op_kwargs={"db_name": db_name,
                   "schema_name": schema_name,
                   "user": db_user,
                   "password": db_pass,
                   "host": db_host,
                   "port": db_port}

    )
