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
>
