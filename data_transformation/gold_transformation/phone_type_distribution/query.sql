SELECT
    label,
    count(*) AS phone_count
FROM blackdiamond_silver.contact_phones
WHERE label IS NOT NULL
GROUP BY label
ORDER BY phone_count DESC