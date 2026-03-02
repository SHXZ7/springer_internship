DAY -2 

> I worked on project DHAP-34, where I built an end-to-end data ingestion pipeline using Apache Airflow and PostgreSQL, fully containerized with Docker.
> 

> First, I selected a completed dataset called customer_care_emails and created the required dataset scaffolding.
> 
> 
> This included a **manifest file**, a **schema contract in YAML**, a **SQL DDL file** to create the target table, and a **sample CSV**.
> 
> This ensured the dataset structure was clearly defined and reproducible.
> 

> Next, I set up a Dockerized Airflow environment using Docker Compose.
> 
> 
> This environment included the Airflow webserver, scheduler, and a PostgreSQL database, all running locally.
> 
> I verified that Airflow was accessible through the browser and that the environment could be started and stopped consistently.
> 

> After that, I implemented an Airflow DAG that loads the CSV data into PostgreSQL.
> 
> 
> The DAG performs four main steps:
> 
> it checks that the CSV file exists, validates the schema against the YAML contract, applies basic data cleaning, and loads the validated data into PostgreSQL.
> 
> I also handled duplicate records so the pipeline can be safely re-run without failing.
> 

> I validated the pipeline by running the DAG end-to-end and confirming that data was successfully written to the PostgreSQL table using pgAdmin.
> 

> Finally, I created detailed documentation and a runbook explaining how to set up the environment, run the pipeline, troubleshoot common issues, and update the dataset in the future.
> 
> 
> This makes the pipeline easy for another engineer to reproduce and maintain.
> 

> Overall, the project demonstrates a complete, production-style data pipeline with orchestration, validation, containerization, and clear documentation.

day -3
build 
Here’s what it does:

1. **Takes a CSV file**
    
    Located at:
    
    ```
    dags/extraction/customer_care_emails.csv
    ```
    
2. **Checks that the file exists**
    
    If missing → DAG fails.
    
3. **Validates schema**
    
    Compares CSV columns against `schema_expected.yaml`.
    
    If mismatch → DAG fails.
    
4. **Cleans / transforms the data**
    - Strips whitespace
    - Casts types
    - Handles nulls
    - (Optionally) adds partition columns like year/month
5. **Converts it to Parquet**
    
    Parquet is:
    
    - Compressed
    - Columnar
    - Efficient for analytics
6. **Uploads the Parquet file to MinIO**
    
    MinIO acts like S3 object storage.
    

So your pipeline is:

```
CSV → Validate → Transform → Parquet → Object Storage
```

And all of this is orchestrated by Airflow inside Docker.

## DAY 4

> I worked on building **Bronze to Silver layer data transformation pipelines** for the data warehouse.
> 
> This involved creating standardized transformation configurations for two datasets: **contact_employments** and **contact_phones**.

> For each dataset, I created a complete transformation package consisting of five key configuration files.
> 
> The **config.yaml** defines the data source (Bronze layer table), target (Silver layer table), and transformation rules for each column.
> 
> The **dag.yaml** specifies the Airflow DAG settings including schedule, retries, timeouts, and processing configuration.
> 
> The **query.sql** implements the actual transformation logic using SQL to clean, normalize, and deduplicate the data.

> The transformation logic includes multiple data quality steps:
> 
> - **String cleaning**: Trimming whitespace, converting to lowercase where appropriate, handling empty strings as nulls
> - **Data validation**: Ensuring required fields like contact_id and organization_id are present and valid
> - **Deduplication**: Using ROW_NUMBER() with PARTITION BY to keep only the most recent and relevant records
> - **Date normalization**: Parsing dates consistently and handling null values properly
> - **Boolean defaults**: Ensuring boolean flags like is_current have proper default values

> I defined the **schema.yaml** for each target table, specifying column names, data types, nullability constraints, and descriptions.
> 
> This creates a clear schema contract that documents what the Silver layer tables should contain.

> Finally, I built a comprehensive **tests.yaml** file with automated data quality tests:
> 
> - Row count validation to ensure data exists
> - NOT NULL checks for required fields
> - Empty string validation for key columns
> - Data freshness checks to ensure timely updates
> - Regex pattern matching for UUID format validation
> - Custom SQL tests for business logic validation
> - Occupation normalization checks to ensure lowercase consistency

> This approach creates **reusable, configuration-driven transformation pipelines** that are easy to maintain and extend.
> 
> Each dataset follows the same pattern, making it simple to onboard new transformations in the future.
> 
> The combination of declarative configuration and SQL-based logic provides both flexibility and clarity.


