import os
import pandas as pd
import numpy as np
import psycopg2
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


default_args = {
    'owner': 'mosula',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}


def process_data(data_name):
    df = pd.read_csv(os.path.join('data', data_name))

    binary_cols = [
        "GENDER",
        "SENIOR_CITIZEN",
        "PARTNER",
        "DEPENDENTS",
        "PHONE_SERVICE",
        "PAPERLESS_BILLING",
        "CHURN"
    ]

    df_processed = df.copy()

    for col in binary_cols:
        df_processed[col] = np.where(df_processed[col] == "yes", 1, 0)
        df_processed[col] = df_processed[col].astype("int32")


def split_data(df):
    pass


def train_model(df):
    pass


def batch_predict(df):
    pass


with DAG(
    default_args=default_args,
    dag_id='fill_data_v11',
    description='Processes and uploads data to the database and then trains the model',
    start_date=datetime(2024, 4, 28),
    schedule_interval='@monthly'
) as dag:
    task1 = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        op_kwargs={'data_name': 'data.csv'}
    )

    task2 = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
    )

    task3 = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    task4 = PythonOperator(
        task_id='batch_predict',
        python_callable=batch_predict,
    )

    task1 >> task2 >> task3 >> task4
