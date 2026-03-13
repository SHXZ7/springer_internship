## Gold Layer Design for BlackDiamond Analytics
# 1. Architecture Overview

**The data pipeline follows a multi-layer architecture designed for scalable analytics.**

Source Systems
      ↓
Raw Layer
      ↓
Silver Layer (Cleaned & normalized ORC data stored in MinIO)
      ↓
Gold Layer (Aggregated datasets optimized for analytics dashboards)

**Layer Responsibilities**

Layer	               Description
Raw	    Ingested source data with minimal transformation
Silver	Cleaned, validated, and normalized data
Gold	Aggregated datasets optimized for analytics dashboards

The Silver layer stores ORC files in MinIO, which are optimized for analytical workloads but may still require scanning large datasets.
The Gold layer pre-aggregates commonly used metrics, allowing dashboards to query smaller datasets and improve performance.

# 2. Silver Data Sources

The Gold layer is built using the following Silver tables:

Silver Table	                              Description
blackdiamond_silver.contact_employments	Cleaned employment information for contacts
blackdiamond_silver.contact_phones	    Cleaned phone number information for contacts

These Silver tables provide validated and normalized data, which serves as the foundation for analytics metrics.

# 3. Gold Layer Design

The Gold layer provides aggregated datasets designed for analytics dashboards.

gold/
└── blackdiamond-metrics/
    contacts_by_org
    contacts_by_occupation
    current_employments
    contacts_with_phone
    phone_type_distribution
    contacts_without_phone

**Metric Definitions**
Metric	                            Analytical Purpose
contacts_by_org	        Analyze the number of contacts associated with each organization
contacts_by_occupation	Understand distribution of contacts across occupations
current_employments	    Identify contacts with active employment
contacts_with_phone	    Measure contact reachability through phone numbers
phone_type_distribution	Analyze usage of different phone types
contacts_without_phone	Identify contacts missing phone information

These metrics represent business-level insights commonly required by dashboards and reporting systems.

# 4. Example Implementations

**Contacts by Organization**

SELECT
    glynac_organization_id,
    COUNT(DISTINCT contact_id) AS contact_count
FROM blackdiamond_silver.contact_employments
GROUP BY glynac_organization_id

This metric provides the number of contacts associated with each organization.

**Contacts With Phone Numbers**

SELECT
    glynac_organization_id,
    COUNT(DISTINCT contact_id) AS contacts_with_phone
FROM blackdiamond_silver.contact_phones
GROUP BY glynac_organization_id

This metric measures how many contacts have phone numbers available for communication.

# 5. Dashboard Use Cases

The Gold layer supports analytics dashboards that answer key business questions such as:

How many contacts exist in each organization?

What occupations are most common among contacts?

How many contacts are currently employed?

What percentage of contacts have phone numbers?

What phone types are most commonly used?

By providing pre-aggregated datasets, dashboards can retrieve insights quickly without scanning large Silver datasets.

# 6. Architecture Diagram

            Source Systems
        (BlackDiamond CRM Data)
                    │
                    ▼
              Raw Layer
     (Ingested source tables)
                    │
                    ▼
              Silver Layer
   Cleaned & Normalized Data (ORC)
           Stored in MinIO
                    │
                    ▼
               Gold Layer
        Aggregated Analytics Tables
                    │
                    ▼
           Analytics Dashboards
        (Business Intelligence Tools)

# 7. Summary

The proposed Gold layer design focuses on:

Providing dashboard-ready aggregated datasets

Reducing query cost on large ORC Silver datasets stored in MinIO

Supporting fast and efficient analytics queries

Maintaining clear separation between data transformation layers

This approach ensures that analytics dashboards can efficiently consume curated and optimized data from the Gold layer.