SELECT
    glynac_organization_id,
    count(DISTINCT contact_id) AS current_employment_count
FROM blackdiamond_silver.contact_employments
WHERE is_current = true
GROUP BY glynac_organization_id
ORDER BY current_employment_count DESC