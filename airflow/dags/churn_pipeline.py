from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

import churn_utils

import os
from dotenv import load_dotenv

load_dotenv()

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")
schema_name = 'churn'

default_args = {
    'owner': 'mosula',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    default_args=default_args,
    dag_id='churn_pipeline_v27',
    description='Processes, uploads data to the database and trains model',
    start_date=datetime(2024, 7, 1),
    schedule_interval='@monthly'
) as dag:
    task1 = PythonOperator(
        task_id='process_data',
        python_callable=churn_utils.process_data,
        op_kwargs={'data_name': 'data.csv',
                   "schema_name": schema_name,
                   "user": db_user,
                   "password": db_pass,
                   "host": db_host,
                   "port": db_port}
    )

    task2 = PythonOperator( 
        task_id='train_model',
        python_callable=churn_utils.split_data_and_train_model,
        op_kwargs={"schema_name": schema_name,
                   "user": db_user,
                   "password": db_pass,
                   "host": db_host,
                   "port": db_port}
    )

    task3 = PythonOperator(
        task_id='batch_predict',
        python_callable=churn_utils.batch_predict,
        op_kwargs={"schema_name": schema_name,
                   "user": db_user,
                   "password": db_pass,
                   "host": db_host,
                   "port": db_port}
    )

    task1 >> task2
    task1 >> task3
