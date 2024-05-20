import os

import json
import pandas as pd
import numpy as np
import catboost as cat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

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


def split_data_and_train_model():
    conn = psycopg2.connect(dbname="churn", user="airflow",
                            password="airflow", host="host.docker.internal")
    cur = conn.cursor()

    df = pd.read_sql("""
    select * from churn_date
    """, con=conn).drop(columns=['id', 'customer_id', 'hist_date'])

    target = 'CHURN'
    test_size = 0.15
    val_size = test_size / (1 - test_size)
    random_state = 0

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
    with open('model_config.json') as file:
        config = json.load(file)

    params = {}
    for key, value in config.items():
        params[key] = value

    model = cat.CatBoostClassifier(**params)
    train_pool = cat.Pool(X_train, y_train, cat_features=cat_columns)
    val_pool = cat.Pool(X_val, y_val, cat_features=cat_columns)

    model.fit(train_pool, eval_set=val_pool)

    metrics = {}
    for key, data in {'train': [X_train, y_train],
                      'val': [X_val, y_val],
                      'test': [X_test, y_test]}.items():
        pred = model.predict_proba(data[0])[:, -1]

        auc = roc_auc_score(data[1], pred)
        acc = accuracy_score(data[1], pred > .5)

        metrics[f'{key}_auc'] = auc
        metrics[f'{key}_acc'] = acc

    sql = f"""
        INSERT INTO models (model_params, model_metrics)
        VALUES ('{str(params).replace("'", '"')}',
                '{str(metrics).replace("'", '"')}')
    """
    cur.execute(sql)
    conn.commit()

    cur.close()
    conn.close()

    model.save_model(os.path.join('models',
                     f'model_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'))


def batch_predict():
    conn = psycopg2.connect(dbname="churn", user="airflow",
                            password="airflow", host="host.docker.internal")
    cur = conn.cursor()

    with open('predict_config.json') as file:
        model_name = json.load(file)['model_name']
    model = cat.CatBoostClassifier()
    model.load_model(os.path.join('models', model_name))

    df = pd.read_sql("""
    select * from churn_date
    """, con=conn).drop(columns=['id', 'customer_id', 'hist_date', 'churn'])

    df['prediction'] = model.predict_proba(df[model.feature_names_])[:, -1]
    df = df.loc[:, ['customer_id', 'hist_date', 'prediction']]
    df['model_name'] = model_name

    columns_str = str(tuple(df.columns)).replace("'", "")
    for _, row in df.iterrows():
        sql = f"""
            INSERT INTO predictions {columns_str}
            VALUES {tuple(row)}
        """
        cur.execute(sql)
    
    conn.commit()
    cur.close()
    conn.close()


with DAG(
    default_args=default_args,
    dag_id='fill_data_v11',
    description='Processes, uploads data to the database and trains model',
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

    task1 >> task2 
    task1 >> task3
