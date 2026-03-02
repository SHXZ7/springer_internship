-- Silver Layer Transformation: Contact Phones
-- Source: blackdiamond.contact_phones_raw
-- Target: blackdiamond_silver.contact_phones
-- Description: Clean and normalize contact phone data (PII)
-- Note: Phone numbers are normalized to digits-only format

SELECT *
FROM (
    SELECT
        -- Contact identifier (trim whitespace, ensure not empty)
        trimBoth(contact_id) AS contact_id,
        
        -- Phone type/category (trim and clean)
        nullIf(trimBoth(phone_type), '') AS phone_type,
        
        -- Phone number - clean and normalize to digits only (remove all non-digits)
        nullIf(
            replaceRegexpAll(trimBoth(phone_number), '[^0-9]', ''),
            ''
        ) AS phone_number,
        
        -- Organization identifier (trim, ensure not empty)
        trimBoth(glynac_organization_id) AS glynac_organization_id,
        
        -- Processing metadata
        processing_date,
        
        -- Deduplication ranking (across all processing dates)
        -- Prefer longer phone numbers (more complete data)
        ROW_NUMBER() OVER (
            PARTITION BY 
                trimBoth(contact_id),
                nullIf(trimBoth(phone_type), ''),
                trimBoth(glynac_organization_id)
            ORDER BY 
                length(replaceRegexpAll(trimBoth(phone_number), '[^0-9]', '')) DESC,
                processing_date DESC
        ) AS rn
        
    FROM blackdiamond.contact_phones_raw
    
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
WHERE rn = 1
    -- Ensure we have a phone number after cleaning
    AND phone_number IS NOT NULL;