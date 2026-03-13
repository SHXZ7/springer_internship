SELECT
    occupation,
    count(DISTINCT contact_id) AS contact_count
FROM blackdiamond_silver.contact_employments
WHERE occupation IS NOT NULL
GROUP BY occupation
ORDER BY contact_count DESC