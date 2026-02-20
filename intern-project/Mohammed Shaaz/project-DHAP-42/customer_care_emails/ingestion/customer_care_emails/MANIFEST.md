# Dataset Manifest — customer_care_emails

## Dataset Name
customer_care_emails

## Source

Local CSV file:
dags/extraction/customer_care_emails.csv

This file contains customer care email records to be ingested
into the data platform.

---

## Pipeline Overview

The Airflow DAG performs the following steps:

1. File existence check  
2. Schema validation against config/schema_expected.yaml  
3. Data transformation (whitespace trimming, type casting, null handling)  
4. Conversion to Parquet format  
5. Upload to MinIO object storage  

If schema validation fails, the DAG will fail and no data will be written.

---

## Target Storage

Object Storage: MinIO  
Bucket: processed-data  
Path:

processed-data/customer_care_emails/

Output format: Parquet

---

## Partitioning Strategy

(Current implementation)

Data is written as Parquet files into:

processed-data/customer_care_emails/

(If partitioning is enabled, data will be organized by year and month derived from a timestamp column.)

---

## Schema Contract

Schema definition is maintained in:

config/schema_expected.yaml

The schema contract defines:
- Column names
- Data types
- Nullability
- Primary key (if applicable)

The pipeline will fail if:
- Required columns are missing
- Unexpected columns exist
- Schema structure does not match

---

## Ownership

Owner: Mohammed Shaaz
Environment: Dockerized Airflow + MinIO  
Execution: Manual trigger via Airflow UI