from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowFailException
from datetime import datetime
import os
import pandas as pd
import yaml
import psycopg2

CSV_PATH = "/opt/airflow/sample_data/customer_care_emails.csv"
SCHEMA_PATH = "/opt/airflow/config/schema_expected.yaml"
DDL_PATH = "/opt/airflow/config/create_table.sql"


def check_file():
    """Task 1: Check if CSV exists in sample_data/"""
    if not os.path.exists(CSV_PATH):
        raise AirflowFailException(f"CSV file not found at {CSV_PATH}")
    print(f"✓ File exists: {CSV_PATH}")


def validate_schema():
    """Task 2: Validate CSV columns vs schema_expected.yaml"""
    df = pd.read_csv(CSV_PATH)

    with open(SCHEMA_PATH, "r") as f:
        schema = yaml.safe_load(f)

    expected_cols = [col["name"] for col in schema["columns"]]
    csv_cols = list(df.columns)

    if expected_cols != csv_cols:
        raise AirflowFailException(
            f"Schema mismatch.\nExpected: {expected_cols}\nFound: {csv_cols}"
        )
    print(f"✓ Schema valid: {len(expected_cols)} columns match")


def transform_data(ti):
    """Task 3: Strip whitespace, handle nulls, skip status == 'done'"""
    df = pd.read_csv(CSV_PATH)

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Filter out rows with email_status == "done"
    if "email_status" in df.columns:
        original_count = len(df)
        df = df[df["email_status"].str.lower() != "done"]
        print(f"✓ Filtered {original_count - len(df)} 'done' rows, {len(df)} remaining")

    # Save cleaned data
    cleaned_path = "/tmp/cleaned_customer_care_emails.csv"
    df.to_csv(cleaned_path, index=False)

    # Pass path to next task via XCom
    ti.xcom_push(key="cleaned_csv_path", value=cleaned_path)
    print(f"✓ Cleaned data saved to {cleaned_path}")


def load_postgres(ti):
    """Task 4: Create table if not exists, insert rows into Postgres"""
    # Get cleaned CSV path from previous task
    cleaned_path = ti.xcom_pull(
        task_ids="transform_data", key="cleaned_csv_path"
    )

    # Connect to Postgres
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
    )
    cur = conn.cursor()

    # Create table if not exists
    with open(DDL_PATH, "r") as f:
        ddl = f.read()
        cur.execute(ddl)
        print("✓ Table created/verified")

    # Load data and insert rows
    df = pd.read_csv(cleaned_path)

    cols = ",".join(df.columns)
    placeholders = ",".join(["%s"] * len(df.columns))
    insert_sql = f"""
        INSERT INTO public.customer_care_emails ({cols})
        VALUES ({placeholders})
        ON CONFLICT (thread_id) DO NOTHING
    """

    row_count = 0
    for _, row in df.iterrows():
        cur.execute(insert_sql, tuple(row))
        row_count += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"✓ Loaded {row_count} rows to public.customer_care_emails (duplicates skipped)")


# DAG Definition
with DAG(
    dag_id="customer_care_emails_ingest",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["dhap-34", "intern"],
) as dag:

    check_file_task = PythonOperator(
        task_id="check_file",
        python_callable=check_file,
    )

    validate_schema_task = PythonOperator(
        task_id="validate_schema",
        python_callable=validate_schema,
    )

    transform_data_task = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data,
    )

    load_postgres_task = PythonOperator(
        task_id="load_postgres",
        python_callable=load_postgres,
    )

    # Pipeline: check → validate → transform → load
    check_file_task >> validate_schema_task >> transform_data_task >> load_postgres_task

