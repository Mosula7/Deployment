import psycopg2
from airflow import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator


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


with DAG(
    dag_id='config_db_v8',
    description='',
    start_date=datetime(2024, 4, 28),
    schedule_interval='@once'
) as dag:
    task1 = PythonOperator(
        task_id='conf',
        python_callable=conf_db
    )
