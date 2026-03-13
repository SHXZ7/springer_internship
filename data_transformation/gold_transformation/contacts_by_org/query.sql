SELECT
    glynac_organization_id,
    count(DISTINCT contact_id) AS contact_count
FROM blackdiamond_silver.contact_employments
GROUP BY glynac_organization_id