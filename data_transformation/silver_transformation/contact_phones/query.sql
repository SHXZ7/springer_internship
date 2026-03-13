SELECT
    contact_id,
    label,
    number,
    glynac_organization_id,
    processing_date
FROM
(
    SELECT
        trimBoth(contact_id) AS contact_id,

        lower(trimBoth(label)) AS label,

        replaceRegexpAll(
            replaceRegexpAll(number,'⟦PH:[^⟧]+⟧',''),
            '[^0-9]',
            ''
        ) AS number,

        trimBoth(toString(glynac_organization_id)) AS glynac_organization_id,

        processing_date,

        ROW_NUMBER() OVER (
            PARTITION BY
                trimBoth(contact_id),
                replaceRegexpAll(
                    replaceRegexpAll(number,'⟦PH:[^⟧]+⟧',''),
                    '[^0-9]',
                    ''
                )
            ORDER BY
                processing_date DESC
        ) AS rn

    FROM blackdiamond.contact_phones_raw
)
WHERE rn = 1
AND contact_id IS NOT NULL
AND number IS NOT NULL
AND length(number) >= 7;