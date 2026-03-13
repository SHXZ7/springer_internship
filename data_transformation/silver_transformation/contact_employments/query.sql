-- Silver Layer Transformation: Contact Employments
-- Source: blackdiamond.contact_employments_raw
-- Target: blackdiamond_silver.contact_employments
-- Description: Clean and normalize employment history data

SELECT *
FROM (
    SELECT
        -- Contact identifier (trim whitespace, ensure not empty)
        trimBoth(contact_id) AS contact_id,
        
        -- Employment details (clean null values and trim)
        nullIf(trimBoth(employer_name), '') AS employer_name,
        nullIf(trimBoth(title), '') AS title,
        
        -- Occupation (normalize to lowercase, clean whitespace)
        nullIf(lower(trimBoth(occupation)), '') AS occupation,
        
        -- Date fields (clean whitespace and empty strings)
        nullIf(trimBoth(start_date), '') AS start_date,
        nullIf(trimBoth(end_date), '') AS end_date,
        
        -- Boolean flag for current employment
        COALESCE(is_current, false) AS is_current,
        
        -- Organization identifier (trim, ensure not empty)
        trimBoth(glynac_organization_id) AS glynac_organization_id,
        
        -- Processing metadata
        processing_date,
        
        -- Deduplication ranking (across all processing dates)
        ROW_NUMBER() OVER (
            PARTITION BY 
                trimBoth(contact_id),
                nullIf(lower(trimBoth(occupation)), ''),
                trimBoth(glynac_organization_id)
            ORDER BY 
                parseDateTimeBestEffort(start_date) DESC NULLS LAST,
                is_current DESC,
                processing_date DESC
        ) AS rn
        
    FROM blackdiamond.contact_employments_raw
    
    -- Data quality filters
    WHERE 
        -- Ensure required fields are present
        contact_id IS NOT NULL
        AND trimBoth(contact_id) != ''
        AND glynac_organization_id IS NOT NULL
        AND trimBoth(glynac_organization_id) != ''
        AND processing_date IS NOT NULL
        
        -- Validate UUID format for contact_id
        AND match(contact_id, '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
        
        -- Validate organization_id format
        AND glynac_organization_id LIKE 'org-%'
)
WHERE rn = 1;