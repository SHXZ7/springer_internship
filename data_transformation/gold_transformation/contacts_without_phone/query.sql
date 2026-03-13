SELECT
    e.glynac_organization_id,
    COUNT(DISTINCT e.contact_id) AS contacts_without_phone
FROM blackdiamond_silver.contact_employments e
LEFT JOIN blackdiamond_silver.contact_phones p
ON e.contact_id = p.contact_id
WHERE p.contact_id IS NULL
GROUP BY e.glynac_organization_id
ORDER BY contacts_without_phone DESC