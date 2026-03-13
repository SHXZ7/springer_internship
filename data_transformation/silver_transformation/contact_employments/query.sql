SELECT
    contact_id,
    employer_name,
    title,
    occupation,
    start_date,
    end_date,
    is_current,
    glynac_organization_id,
    processing_date
FROM
(
    SELECT
        contact_id,
        employer_name,
        title,
        occupation,
        start_date,
        end_date,
        is_current,
        glynac_organization_id,
        processing_date,

        ROW_NUMBER() OVER (
            PARTITION BY contact_id, occupation, start_date
            ORDER BY is_current DESC, processing_date DESC
        ) AS rn
    FROM
    (
        SELECT
            trimBoth(contact_id) AS contact_id,

            nullIf(trimBoth(toString(employer_name)), '') AS employer_name,
            nullIf(trimBoth(toString(title)), '') AS title,

            nullIf(
                lower(
                    trimBoth(
                        replaceRegexpAll(occupation,'⟦[pP]:[^⟧]+⟧','')
                    )
                ),
            '') AS occupation,

            start_date,
            toDateOrNull(toString(end_date)) AS end_date,
            COALESCE(is_current,false) AS is_current,
            trimBoth(toString(glynac_organization_id)) AS glynac_organization_id,
            processing_date

        FROM blackdiamond.contact_employments_raw
    ) cleaned
) dedup
WHERE rn = 1
AND occupation IS NOT NULL
AND glynac_organization_id != 'org_pii_masking';