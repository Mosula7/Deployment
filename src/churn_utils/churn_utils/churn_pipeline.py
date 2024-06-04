import os

import json
import pandas as pd
import numpy as np
import catboost as cat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import psycopg2
from datetime import datetime


def conf_db():
    conn = psycopg2.connect(dbname="postgres", user="airflow",
                            password="airflow", host="host.docker.internal")
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    schema_name = "churn"
    cur.execute(f"""CREATE DATABASE {schema_name};""")
    conn.commit()

    cur.close()
    conn.close()

    conn = psycopg2.connect(dbname="churn", user="airflow", password="airflow",
                            host="host.docker.internal")
    cur = conn.cursor()

    sql_create_churn_data = """
        CREATE TABLE IF NOT EXISTS churn_data (
            id  SERIAL PRIMARY KEY,
            hist_date DATE default current_date,
            customer_id  VARCHAR(32),
            GENDER INT,
            SENIOR_CITIZEN INT,
            PARTNER INT,
            DEPENDENTS INT,
            TENURE INT,
            PHONE_SERVICE INT,
            MULTIPLE_LINES VARCHAR(32),
            INTERNET_SERVICE VARCHAR(32),
            ONLINE_SECURITY VARCHAR(32),
            ONLINE_BACKUP VARCHAR(32),
            DEVICE_PROTECTION VARCHAR(32),
            TECH_SUPPORT VARCHAR(32),
            STREAMING_TV VARCHAR(32),
            STREAMING_MOVIES VARCHAR(32),
            CONTRACT VARCHAR(32),
            PAPERLESS_BILLING INT,
            PAYMENT_METHOD VARCHAR(32),
            MONTHLY_CHARGES FLOAT,
            TOTAL_CHARGES FLOAT,
            CHURN INT
        );
        """

    sql_create_models = """
        CREATE TABLE IF NOT EXISTS churn_models (
            model_id SERIAL PRIMARY KEY,
            train_date date default current_date,
            model_name VARCHAR(255) NOT NULL,
            model_params JSONB,
            model_metrics JSONB
        );
        """

    sql_create_predictions = """
        CREATE TABLE IF NOT EXISTS churn_predictions (
            prediction_id SERIAL PRIMARY KEY,
            customer_id VARCHAR(32),
            model_name VARCHAR(32),
            hist_date date default current_date,
            prediction FLOAT NOT NULL,
            flag VARCHAR(32)
        );
        """

    for sql_statement in [
        sql_create_churn_data,
        sql_create_models,
        sql_create_predictions
    ]:
        cur.execute(sql_statement)

    conn.commit()

    cur.close()
    conn.close()


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
    df_processed['GENDER'] = np.where(df_processed['GENDER'] == 'female', 1, 0)

    conn = psycopg2.connect(dbname="churn", user="airflow", password="airflow",
                            host="host.docker.internal")
    cur = conn.cursor()

    columns_str = str(tuple(df_processed.columns)).replace("'", "")
    for _, row in df_processed.iterrows():
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
    select * from churn_data
    """, con=conn).drop(columns=['id', 'customer_id', 'hist_date'])

    target = 'churn'
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
        'multiple_lines',
        'internet_service',
        'online_security',
        'online_backup',
        'device_protection',
        'tech_support',
        'streaming_tv',
        'streaming_movies',
        'contract',
        'payment_method'
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
    model_name = f'model_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'

    sql = f"""
        INSERT INTO churn_models (model_params, model_name, model_metrics)
        VALUES ('{str(params).replace("'", '"')}',
                '{model_name}',
                '{str(metrics).replace("'", '"')}')
    """
    cur.execute(sql)
    conn.commit()

    cur.close()
    conn.close()

    model.save_model(os.path.join('models', model_name))


def batch_predict():
    conn = psycopg2.connect(dbname="churn", user="airflow",
                            password="airflow", host="host.docker.internal")
    cur = conn.cursor()

    with open('predict_config.json') as file:
        model_name = json.load(file)['model_name']
    model = cat.CatBoostClassifier()
    model.load_model(os.path.join('models', model_name))

    df = pd.read_sql("""
    select * from churn_data
    """, con=conn)

    df.columns = [col.upper() for col in df.columns]

    df['prediction'] = model.predict_proba(df[model.feature_names_])[:, -1]
    df = df.loc[:, ['CUSTOMER_ID', 'prediction']]
    df['model_name'] = model_name
    df['flag'] = 'batch'

    columns_str = str(tuple(df.columns)).replace("'", "")
    for _, row in df.iterrows():
        sql = f"""
            INSERT INTO churn_predictions {columns_str}
            VALUES {tuple(row)}
        """
        cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()
