# Customer Care Emails – Airflow Ingestion Pipeline

## 1. Dataset Overview

**Dataset name:** `customer_care_emails`
**Purpose:**
This dataset contains customer support email threads used for downstream analytics and processing.
The pipeline ingests a local CSV file, validates it against a schema contract, applies basic transformations, and loads clean records into PostgreSQL.

**CSV Location:**
```
sample_data/customer_care_emails.csv
```

**Target PostgreSQL Table:**
```
public.customer_care_emails
```

The table structure is defined in:
```
config/create_table.sql
```

---

## 2. Repository Structure

```
customer_care_emails/
├── extraction/
│   └── customer_care_emails/
│       ├── dags/
│       │   └── customer_care_emails_ingest.py
│       └── README.md (this file)
├── config/
│   ├── create_table.sql
│   └── schema_expected.yaml
├── sample_data/
│   └── customer_care_emails.csv
├── docker/
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── .env
├── docs/
│   └── README.md
└── MANIFEST.md
```

---

## 3. Environment Variables

Environment variables are loaded via Docker Compose from `.env`.

Location:
```
docker/.env
```

Required variables:

```env
# Airflow metadata DB
AIRFLOW_DB_USER=airflow
AIRFLOW_DB_PASSWORD=airflow
AIRFLOW_DB_NAME=airflow

# Target Postgres (local)
PG_HOST=local_postgres
PG_PORT=5432
PG_DB=customer_care
PG_USER=postgres
PG_PASSWORD=postgres
```

⚠️ **Security Note:**
- Do not commit `.env` files with real credentials
- Create a `.env.sample` template for documentation

---

## 4. How to Run the Pipeline

### Step 1: Start Airflow & Postgres

From the `docker/` directory:

```bash
cd docker
docker compose up -d
```

Verify containers:

```bash
docker compose ps
```

Expected output:
- `airflow-db` (Airflow metadata database)
- `local_postgres` (Target database for customer care data)
- `airflow-webserver` (Airflow UI)
- `airflow-scheduler` (DAG scheduler)

Airflow UI should be available at:
```
http://localhost:8080
```

### Step 2: Access Airflow UI

Login credentials (configured in docker-compose.yml):
- **Username:** `admin`
- **Password:** `admin`

Navigate to **DAGs → customer_care_emails_ingest**

### Step 3: Trigger the DAG

**Option A: Via UI**
- Click on the DAG name
- Click the "Play" button (▶) to trigger

**Option B: Via CLI**
```bash
docker compose exec airflow-webserver airflow dags trigger customer_care_emails_ingest
```

---

## 5. DAG Overview

**DAG Name:** `customer_care_emails_ingest`

### Task Flow
```
check_file
   ↓
validate_schema
   ↓
transform_data
   ↓
load_postgres
```

### Task Responsibilities

#### `check_file`
- Verifies the CSV exists at `/opt/airflow/sample_data/customer_care_emails.csv`
- Fails DAG if file is missing
- **Location:** customer_care_emails_ingest.py:15-19

#### `validate_schema`
- Compares CSV columns against `schema_expected.yaml`
- Validates column names and order
- Fails DAG on schema mismatch
- **Location:** customer_care_emails_ingest.py:22-38

#### `transform_data`
- Strips whitespace from all string columns
- Filters out rows where `email_status == "done"` (case-insensitive)
- Saves cleaned data to `/tmp/cleaned_customer_care_emails.csv`
- Passes file path to next task via XCom
- **Location:** customer_care_emails_ingest.py:41-61

#### `load_postgres`
- Creates table using `create_table.sql` (IF NOT EXISTS)
- Inserts cleaned records into PostgreSQL
- **Uses `ON CONFLICT (thread_id) DO NOTHING`** for idempotency
- Skips duplicate rows automatically
- **Location:** customer_care_emails_ingest.py:64-105

---

## 6. Troubleshooting Guide

### ❌ DAG Fails at `check_file`

**Cause:** CSV not found
**Fix:**
1. Ensure file exists at `sample_data/customer_care_emails.csv`
2. Check Docker volume mounts in `docker-compose.yml`:
   ```yaml
   volumes:
     - ../sample_data:/opt/airflow/sample_data
   ```
3. Restart containers if volumes were updated:
   ```bash
   docker compose down
   docker compose up -d
   ```

### ❌ Schema Validation Failure

**Cause:** CSV schema does not match `schema_expected.yaml`
**Fix:**
1. Compare CSV headers with YAML column definitions
2. Check column order (must match exactly)
3. Update YAML only if schema change is intentional
4. Reference: `config/schema_expected.yaml`

### ❌ Database Connection Errors

**Cause:** Invalid credentials or wrong host/port
**Fix:**
1. Verify `.env` values match docker-compose.yml
2. Confirm `local_postgres` container is running:
   ```bash
   docker compose ps local_postgres
   ```
3. Check environment variables are propagated:
   ```bash
   docker compose exec airflow-scheduler env | grep PG_
   ```
4. **Note:** Use `local_postgres` as host (container name), not `localhost`

### ❌ Duplicate Key / Primary Key Errors

**Status:** ✅ **FIXED**
**Previous Cause:** Re-running DAG with existing data caused `UniqueViolation` errors

**Current Solution:**
The `load_postgres` task now uses:
```sql
INSERT INTO public.customer_care_emails (...)
VALUES (...)
ON CONFLICT (thread_id) DO NOTHING
```

This makes the DAG **idempotent** – you can safely rerun without errors. Duplicate `thread_id` values are skipped automatically.

### ❌ Logs Show 403 Forbidden or Session Errors

**Status:** ✅ **FIXED**
**Previous Cause:** Airflow components using different `SECRET_KEY`

**Current Solution:**
Both `airflow-webserver` and `airflow-scheduler` now use:
```yaml
AIRFLOW__WEBSERVER__SECRET_KEY: "internship-secret-key"
```

If you still see issues:
```bash
docker compose down
docker compose up -d
```

### ❌ "No such file or directory" Errors

**Cause:** Files not mounted correctly or paths incorrect
**Fix:**
1. Verify volume mounts in `docker-compose.yml`:
   ```yaml
   volumes:
     - ../sample_data:/opt/airflow/sample_data
     - ../config:/opt/airflow/config
   ```
2. Check file paths in DAG match mounted locations:
   ```python
   CSV_PATH = "/opt/airflow/sample_data/customer_care_emails.csv"
   SCHEMA_PATH = "/opt/airflow/config/schema_expected.yaml"
   DDL_PATH = "/opt/airflow/config/create_table.sql"
   ```

---

## 7. Runbook

### Updating the Schema

1. Update `config/schema_expected.yaml`
2. Update `config/create_table.sql` to match
3. Test DAG locally before committing
4. Update `MANIFEST.md` if table structure changes

### Loading a New CSV Drop

1. Replace or update CSV under `sample_data/`
2. Ensure schema compatibility (run validate_schema task)
3. Trigger DAG manually or wait for schedule
4. Verify records in PostgreSQL:
   ```sql
   SELECT COUNT(*) FROM public.customer_care_emails;
   ```

### Resetting DAG Runs

**Option A: From Airflow UI**
- Navigate to DAG → Clear all failed tasks
- Or delete DAG runs and re-trigger

**Option B: Clear Table and Rerun**
```sql
TRUNCATE TABLE public.customer_care_emails;
```
Then re-trigger DAG from Airflow UI

### Viewing Logs

**Via Airflow UI:**
- Click on task → View Logs

**Via Docker:**
```bash
docker compose logs airflow-scheduler
docker compose logs airflow-webserver
```

---

## 8. Key Implementation Details

### Idempotency
The pipeline is designed to be **idempotent** – running it multiple times with the same data produces the same result:
- `CREATE TABLE IF NOT EXISTS` prevents table creation errors
- `ON CONFLICT DO NOTHING` skips duplicate records
- Data transformations are deterministic

### Data Quality Checks
1. **File existence check** – prevents processing if source is missing
2. **Schema validation** – enforces data contract
3. **Whitespace normalization** – ensures clean data
4. **Status filtering** – skips already processed records

### Performance Considerations
- Row-by-row insertion (current implementation)
- For large datasets (>100K rows), consider bulk insert with `COPY` or `executemany()`
- CSV size: ~1.4MB (acceptable for current approach)

---

## 9. Dependencies

Defined in `docker/requirements.txt`:
```
pandas
psycopg2-binary
pyyaml
```

These are installed automatically when containers start.

---

## 10. Pre-Commit Checklist (For New Changes)

Before committing changes:

- [ ] `MANIFEST.md` updated
- [ ] `schema_expected.yaml` validated
- [ ] `create_table.sql` aligned with schema
- [ ] Sample CSV added/updated
- [ ] DAG runs successfully end-to-end
- [ ] No secrets committed (`.env` excluded)
- [ ] README.md updated if workflow changes

---

## 11. Reproducibility

A new engineer should be able to:

1. Clone the repo
2. Navigate to `docker/` directory
3. Create `.env` file with variables (see section 3)
4. Run `docker compose up -d`
5. Trigger DAG from Airflow UI
6. See data loaded into PostgreSQL

✅ **End-to-end reproducible.**

---

## 12. Quick Reference

| Item | Value |
|------|-------|
| Airflow UI | http://localhost:8080 |
| Login | admin / admin |
| DAG ID | customer_care_emails_ingest |
| Target DB | local_postgres:5432 |
| Target Schema | public |
| Target Table | customer_care_emails |
| Primary Key | thread_id |
| Schedule | Manual (None) |

---

## 13. Contact & Support

For questions or issues with this pipeline:
- Check Airflow task logs first
- Review troubleshooting section (Section 6)
- Consult `MANIFEST.md` for dataset context

---

**Last Updated:** 2026-02-07
**Maintained by:** Mohammed Shaaz
**Project:** DHAP-34
