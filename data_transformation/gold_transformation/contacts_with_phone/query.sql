SELECT
    glynac_organization_id,
    count(DISTINCT contact_id) AS contacts_with_phone
FROM blackdiamond_silver.contact_phones
GROUP BY glynac_organization_id
ORDER BY contacts_with_phone DESC