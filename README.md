# 🗂️ Data Engineering Internship — Documentation Log

> **Duration:** 3 Months &nbsp;|&nbsp; **Domain:** Data Engineering &nbsp;|&nbsp; **Role:** Data Engineering Intern  
> A structured log of weekly deliverables, technical implementations, and learnings across a production-grade data engineering internship.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Month 1 — Pipeline Foundations](#month-1--pipeline-foundations)
- [Month 2 — Lakehouse Architecture](#month-2--lakehouse-architecture)
- [Month 3 — Gold Layer & Analytics](#month-3--gold-layer--analytics)
- [Key Takeaways](#key-takeaways)

---

## Overview

This repository documents my 3-month data engineering internship, organized week by week. Each entry covers the project context, what was built, the technical approach, and outcomes. The work spans end-to-end data pipeline development — from raw ingestion through orchestration, transformation, quality validation, and analytics-ready delivery.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | Apache Airflow |
| Storage | PostgreSQL, MinIO (S3-compatible) |
| Containerization | Docker, Docker Compose |
| Transformation | SQL, Python, Parquet |
| Data Quality | YAML schema contracts, custom test suites |
| Monitoring | pgAdmin, Airflow UI |
| Formats | CSV → Parquet → PostgreSQL |

---

## Month 1 — Pipeline Foundations

> **Focus:** Building a production-style ingestion pipeline from scratch with orchestration, validation, and containerization.

---

### Week 1 — Dataset Scaffolding & Environment Setup

**Project:** `DHAP-34` — End-to-end data ingestion pipeline

**What was built:**
- Selected the `customer_care_emails` dataset and created the full dataset scaffold
- Defined a **manifest file**, **schema contract (YAML)**, **SQL DDL** for the target table, and a **sample CSV**
- Set up a Dockerized Airflow environment using Docker Compose with:
  - Airflow webserver
  - Airflow scheduler
  - PostgreSQL database (all running locally)
- Verified Airflow was accessible via browser and environment could be started/stopped consistently

**Outcome:** A reproducible, fully containerized local development environment with a clearly defined dataset structure.

---

### Week 2 — Airflow DAG Implementation & Pipeline Validation

**Project:** `DHAP-34` continued

**What was built:**
- Implemented an Airflow DAG that loads CSV data into PostgreSQL across 4 steps:
  1. **File existence check** — DAG fails fast if CSV is missing
  2. **Schema validation** — Columns validated against the YAML contract
  3. **Data cleaning** — Whitespace stripping, type casting, null handling
  4. **PostgreSQL load** — Duplicate-safe upsert for idempotent re-runs
- Validated the full pipeline end-to-end using pgAdmin to confirm data was written correctly
- Created detailed **documentation and a runbook** covering setup, execution, troubleshooting, and future dataset updates

**Outcome:** A fully functional, idempotent ingestion pipeline with clear runbook documentation ready for handoff to another engineer.

---

### Week 3 — Object Storage Integration (CSV → Parquet → MinIO)

**What was built:**
- Extended the pipeline to convert validated CSV data into **Parquet format** before storage
- Integrated **MinIO** as an S3-compatible object storage layer
- The updated pipeline flow:

```
CSV → File Check → Schema Validation → Clean & Transform → Parquet → MinIO Upload
```

- Parquet chosen for its columnar compression and analytics efficiency
- All steps orchestrated by Airflow inside Docker

**Key design decisions:**
- Schema contract in `schema_expected.yaml` validated before any transformation
- Partition columns (year/month) optionally added during transformation
- MinIO acts as the Bronze landing zone for raw-but-validated data

**Outcome:** Analytics-optimized object storage pipeline with full orchestration and validation gates.

---

### Week 4 — Month 1 Review & Pipeline Hardening

**Focus:** Stabilization, documentation cleanup, and pipeline reliability review

**What was done:**
- End-to-end pipeline re-run tests across multiple scenarios (missing file, schema mismatch, duplicate data)
- Confirmed all failure modes exit cleanly with descriptive error messages
- Reviewed runbook with team and incorporated feedback
- Finalized Month 1 deliverables for handoff

**Outcome:** Production-hardened pipeline with validated failure handling and complete documentation.

---

## Month 2 — Lakehouse Architecture

> **Focus:** Building Bronze-to-Silver transformation layers with configuration-driven pipelines and automated data quality testing.

---

### Week 5 — Bronze → Silver: `contact_employments` Transformation

**What was built:**

A complete transformation package for the `contact_employments` dataset with 5 configuration files:

| File | Purpose |
|---|---|
| `config.yaml` | Source (Bronze), target (Silver), column-level transformation rules |
| `dag.yaml` | Airflow DAG settings — schedule, retries, timeouts |
| `query.sql` | SQL transformation logic — cleaning, normalization, deduplication |
| `schema.yaml` | Silver table schema contract — column names, types, nullability |
| `tests.yaml` | Automated data quality test suite |

**Transformation logic implemented:**
- String cleaning: whitespace trimming, lowercase normalization, empty → null
- Required field validation: `contact_id`, `organization_id`
- Deduplication via `ROW_NUMBER() PARTITION BY` (most recent record wins)
- Date normalization with null-safe parsing
- Boolean defaults for `is_current` flag

**Outcome:** A reusable, config-driven Bronze → Silver pipeline for employment data.

---

### Week 6 — Bronze → Silver: `contact_phones` Transformation

**What was built:**

Applied the same standardized transformation pattern to the `contact_phones` dataset:

- `config.yaml`, `dag.yaml`, `query.sql`, `schema.yaml`, `tests.yaml` — all defined
- Phone-specific transformation rules: label normalization, null-safe handling
- Schema contract defining output Silver table structure with type and nullability constraints

**Data quality tests defined in `tests.yaml`:**
- Row count validation (data existence check)
- NOT NULL enforcement on required fields
- Empty string validation for key columns
- Data freshness checks for timely pipeline updates
- UUID format validation via regex pattern matching
- Custom SQL tests for business logic
- Occupation normalization checks (lowercase consistency)

**Outcome:** Fully tested Silver layer for phone data following the same reusable pattern as Week 5.

---

### Week 7 — Configuration-Driven Pipeline Standardization

**Focus:** Making the Bronze → Silver pattern reusable across all future datasets

**What was done:**
- Formalized the 5-file configuration standard as the team's transformation template
- Documented the pattern: how to onboard a new dataset using the same structure
- Reviewed both `contact_employments` and `contact_phones` transformations for consistency
- Tested pipelines for edge cases: nulls, schema drift, duplicate handling

**Key insight:** Declarative configuration (YAML) + SQL transformation logic = flexibility with clarity. Any engineer can onboard a new dataset without writing orchestration code from scratch.

**Outcome:** A documented, reusable transformation framework ready for scale.

---

### Week 8 — Month 2 Review & Silver Layer Sign-off

**Focus:** End-to-end Silver layer validation and quality gate review

**What was done:**
- Full pipeline run: Bronze ingestion → Silver transformation → quality test execution
- Verified all `tests.yaml` checks pass for both datasets
- Reviewed schema contracts against actual Silver table state in PostgreSQL
- Documentation updated to reflect final Silver layer structure

**Outcome:** Two production-ready Silver tables (`blackdiamond_silver.contact_employments`, `blackdiamond_silver.contact_phones`) with automated quality coverage.

---

## Month 3 — Gold Layer & Analytics

> **Focus:** Silver-to-Gold transformations producing pre-aggregated, analytics-ready datasets for business reporting.

---

### Week 9 — Gold Layer Design & First Aggregations

**What was built:**

Designed the Gold layer architecture and implemented the first three Gold datasets:

| Gold Dataset | Logic |
|---|---|
| `contacts_by_org` | Distinct contact count per organization |
| `contacts_by_occupation` | Occupation-wise contact distribution |
| `current_employments` | Active employment count (`is_current = true`) by org |

Each Gold dataset packaged with:
- `query.sql` — aggregation logic
- `schema.yaml` — output schema contract
- `dag.yaml` — orchestration configuration
- `tests.yaml` — `not_null` checks on all dimension and metric columns

**Outcome:** Three analytics-ready Gold tables with full orchestration and quality coverage.

---

### Week 10 — Gold Layer: Phone Analytics & Contact Coverage

**What was built:**

Three additional Gold datasets focused on phone coverage metrics:

| Gold Dataset | Logic |
|---|---|
| `contacts_with_phone` | Count of contacts that have a phone number |
| `contacts_without_phone` | Contacts missing phone numbers (LEFT JOIN pattern) |
| `phone_type_distribution` | Distribution of phone labels/types across contacts |

**Design decision:** `contacts_without_phone` uses a LEFT JOIN with NULL filter rather than a subquery — more performant and readable at scale.

**Outcome:** Complete phone coverage analytics surfaced at the Gold layer, ready for dashboard consumption.

---

### Week 11 — Pipeline Integration & End-to-End Testing

**Focus:** Full Bronze → Silver → Gold pipeline integration run

**What was done:**
- Triggered the complete 3-layer pipeline in sequence via Airflow
- Validated row counts and metric values across all 6 Gold tables
- Confirmed all `tests.yaml` quality checks pass end-to-end
- Benchmarked query performance on Gold tables vs raw Bronze queries
- Documented the full data lineage: source CSV → Bronze → Silver → Gold

**Outcome:** A fully validated, end-to-end lakehouse pipeline with 6 production-ready Gold datasets.

---

### Week 12 — Final Documentation, Handoff & Retrospective

**Focus:** Closing the internship with clean handoff artefacts

**What was done:**
- Finalized all runbooks across Month 1–3
- Created a data lineage diagram documenting the full pipeline flow
- Wrote onboarding guide for the next engineer: how to add a new dataset through all 3 layers
- Conducted retrospective: what worked, what would be improved with more time
- Final presentation of the complete pipeline to the team

**Outcome:** Full project handoff with documentation, lineage, and onboarding materials complete.

---

## Key Takeaways

| Area | Learning |
|---|---|
| **Orchestration** | Airflow DAGs with clear task dependencies make pipelines debuggable and observable |
| **Validation** | Schema contracts at every layer (YAML) catch issues before they reach production |
| **Idempotency** | Duplicate-safe pipeline design is non-negotiable for reliable re-runs |
| **Configuration-driven design** | Separating SQL logic from orchestration config makes pipelines scalable |
| **Layered architecture** | Bronze / Silver / Gold separation improves both data quality and query performance |
| **Documentation** | Runbooks written as you build are far better than written after the fact |

---

<div align="center">

**3 months · 12 weeks · Bronze → Silver → Gold**  
Built with Apache Airflow · PostgreSQL · Docker · MinIO · Python · SQL

</div>
