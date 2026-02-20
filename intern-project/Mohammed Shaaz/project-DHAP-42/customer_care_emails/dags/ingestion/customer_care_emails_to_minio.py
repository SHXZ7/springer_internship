import os
import yaml
import pandas as pd
import boto3
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator # type: ignore
from airflow.exceptions import AirflowFailException

DATASET_NAME = "customer_care_emails"
CSV_PATH = "/opt/airflow/dags/extraction/customer_care_emails.csv"
SCHEMA_PATH = "/opt/airflow/ingestion/customer_care_emails/config/schema_expected.yaml"
LOCAL_PARQUET_PATH = f"/tmp/{DATASET_NAME}"

MINIO_BUCKET = os.getenv("MINIO_BUCKET")
MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")


def check_file():
    if not os.path.exists(CSV_PATH):
        raise AirflowFailException("CSV file not found!")


def validate_schema():
    df = pd.read_csv(CSV_PATH)
    with open(SCHEMA_PATH, "r") as f:
        schema = yaml.safe_load(f)

    expected_columns = [col["name"] for col in schema["columns"]]
    actual_columns = df.columns.tolist()

    if set(expected_columns) != set(actual_columns):
        raise AirflowFailException(
            f"Schema mismatch! Expected {expected_columns}, got {actual_columns}"
        )


def transform_data():
    df = pd.read_csv(CSV_PATH)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Convert timestamp column and create partitioning columns
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Check for any parsing failures
    null_timestamps = df["timestamp"].isna().sum()
    if null_timestamps > 0:
        print(f"WARNING: {null_timestamps} rows have invalid timestamps and will be dropped")
        df = df.dropna(subset=["timestamp"])
    
    df["year"] = df["timestamp"].dt.year.astype("Int64")
    df["month"] = df["timestamp"].dt.month.astype("Int64")
    
    print(f"Data shape: {df.shape}")
    print(f"Year values: {df['year'].unique()}")
    print(f"Month values: {df['month'].unique()}")

    df.to_pickle("/tmp/transformed.pkl")


def convert_to_parquet():
    df = pd.read_pickle("/tmp/transformed.pkl")

    # Verify partition columns exist
    if "year" not in df.columns or "month" not in df.columns:
        raise AirflowFailException("Partition columns (year, month) not found in dataframe!")
    
    print(f"Converting {len(df)} rows to partitioned parquet")
    print(f"Partition columns: year={df['year'].unique()}, month={df['month'].unique()}")

    # Write to partitioned parquet directory
    os.makedirs(LOCAL_PARQUET_PATH, exist_ok=True)
    df.to_parquet(
        LOCAL_PARQUET_PATH,
        engine="pyarrow",
        partition_cols=["year", "month"],
        index=False
    )
    
    print(f"Parquet written to {LOCAL_PARQUET_PATH}")


def upload_to_minio():
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )

    if not os.path.isdir(LOCAL_PARQUET_PATH):
        raise AirflowFailException(f"Parquet directory not found: {LOCAL_PARQUET_PATH}")
    
    uploaded_count = 0
    # Upload partitioned parquet files
    for root, dirs, files in os.walk(LOCAL_PARQUET_PATH):
        for file in files:
            if file.endswith('.parquet'):
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, LOCAL_PARQUET_PATH)
                s3_path = f"{DATASET_NAME}/{relative_path}"
                
                print(f"Uploading {local_path} to s3://{MINIO_BUCKET}/{s3_path}")
                s3.upload_file(local_path, MINIO_BUCKET, s3_path)
                uploaded_count += 1
    
    print(f"Successfully uploaded {uploaded_count} parquet files to MinIO")


default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
}

with DAG(
    dag_id=f"{DATASET_NAME}_to_minio",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    t1 = PythonOperator(task_id="check_file", python_callable=check_file)
    t2 = PythonOperator(task_id="validate_schema", python_callable=validate_schema)
    t3 = PythonOperator(task_id="transform_data", python_callable=transform_data)
    t4 = PythonOperator(task_id="convert_to_parquet", python_callable=convert_to_parquet)
    t5 = PythonOperator(task_id="upload_to_minio", python_callable=upload_to_minio)

    t1 >> t2 >> t3 >> t4 >> t5