import os

import pandas as pd
import numpy as np
import catboost as cat
from sklearn.model_selection import train_test_split

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
    conn = psycopg2.connect(dbname="churn", user="airflow", password="airflow",
                            host="host.docker.internal")
    cur = conn.cursor()

    columns_str = str(tuple(df.columns)).replace("'", "")
    for _, row in df.iterrows():
        sql = f"""
            INSERT INTO churn_data {columns_str}
            VALUES {tuple(row)}
        """
        cur.execute(sql)

    conn.commit()
    cur.close()
    conn.close()


def split_data_and_train_model(df: pd.DataFrame, target: str, test_size: float,
                               val_size: float = None, random_state: int = 0):
    """
    returns (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if not val_size:
        val_size = test_size / (1 - test_size)

    train_val, test = train_test_split(df,
                                       test_size=test_size,
                                       stratify=df[target],
                                       random_state=random_state)
    train, val = train_test_split(train_val,
                                  test_size=val_size,
                                  stratify=train_val[target],
                                  random_state=random_state)

    X_train = train[train.columns.drop(target)]
    X_val = val[val.columns.drop(target)]
    X_test = test[test.columns.drop(target)]

    y_train = train[target]
    y_val = val[target]
    y_test = test[target]
    cat_columns = [
        'MULTIPLE_LINES',
        'INTERNET_SERVICE',
        'ONLINE_SECURITY',
        'ONLINE_BACKUP',
        'DEVICE_PROTECTION',
        'TECH_SUPPORT',
        'STREAMING_TV',
        'STREAMING_MOVIES',
        'CONTRACT',
        'PAYMENT_METHOD'
    ]
    model = cat.CatBoostClassifier(early_stopping_rounds=20)
    train_pool = cat.Pool(X_train, y_train, cat_features=cat_columns)
    val_pool = cat.Pool(X_val, y_val, cat_features=cat_columns)


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
        python_callable=split_data_and_train_model,
    )

    task3 = PythonOperator(
        task_id='batch_predict',
        python_callable=batch_predict,
    )

    task1 >> task2 >> task3
