from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

import churn_utils

default_args = {
    'owner': 'mosula',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    default_args=default_args,
    dag_id='fill_data_v26',
    description='Processes, uploads data to the database and trains model',
    start_date=datetime(2024, 4, 28),
    schedule_interval='@monthly'
) as dag:
    task1 = PythonOperator(
        task_id='process_data',
        python_callable=churn_utils.process_data,
        op_kwargs={'data_name': 'data.csv'}
    )

    task2 = PythonOperator(
        task_id='train_model',
        python_callable=churn_utils.split_data_and_train_model,
    )

    task3 = PythonOperator(
        task_id='batch_predict',
        python_callable=churn_utils.batch_predict,
    )

    task1 >> task2
    task1 >> task3
