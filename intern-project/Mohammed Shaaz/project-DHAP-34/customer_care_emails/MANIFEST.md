# Dataset Manifest

Dataset Name:
customer_care_emails

Local CSV Path:
airflow-dags/extraction/customer_care_emails/sample_data/customer_care_emails.csv

Target Table:
public.customer_care_emails

Description:
Customer care email records ingested from a locally downloaded CSV
(original source: SharePoint). The dataset is validated against an
expected schema and loaded into PostgreSQL via Airflow.
