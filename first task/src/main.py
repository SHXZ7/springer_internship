from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytz


_DEFAULT_DATETIME = pd.Timestamp("1970-01-01 00:00:00")


def _excel_safe_datetime(value: object) -> object:
    # Excel cannot store timezone-aware datetimes.
    if isinstance(value, pd.Timestamp):
        if value.tz is not None:
            return value.tz_localize(None)
        return value
    # Handle python datetime with tzinfo, if any appear.
    tzinfo = getattr(value, "tzinfo", None)
    if tzinfo is not None:
        return value.replace(tzinfo=None)
    return value


def _to_snake_case(name: str) -> str:
    s = name.strip()
    s = s.replace("-", "_").replace(" ", "_")
    # Insert underscore before capitals (simple CamelCase handling)
    out: list[str] = []
    for i, ch in enumerate(s):
        if ch.isupper() and i > 0 and (s[i - 1].islower() or (i + 1 < len(s) and s[i + 1].islower())):
            out.append("_")
        out.append(ch.lower())
    snake = "".join(out)
    # Collapse repeated underscores
    while "__" in snake:
        snake = snake.replace("__", "_")
    return snake.strip("_")


def _is_datetime_like(series: pd.Series, column_name: str) -> bool:
    """
    Heuristic: treat as datetime if pandas already thinks it is datetime, or if
    the column name suggests time/date and a substantial portion parses.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    name_hint = column_name.lower()
    if any(token in name_hint for token in ("_at", "date", "time", "timestamp", "created", "updated", "expired")):
        parsed = pd.to_datetime(series, errors="coerce", utc=False)
        # If at least 80% of non-null values parse, call it datetime-like.
        non_null = series.notna().sum()
        if non_null == 0:
            return False
        return parsed.notna().sum() / non_null >= 0.8

    return False


def clean_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Cleaning rules (before joins):
    - standardize column names to snake_case
    - trim strings
    - convert timestamp/date-like columns to datetime
    - drop rows missing required keys
    - ensure final table has no nulls (fill remaining with sensible defaults)
    """
    out = df.copy()

    # Standardize column names
    out.columns = [_to_snake_case(c) for c in out.columns]

    # Trim strings early (before parsing datetimes)
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].astype("string").str.strip()
            # Keep real missing values as <NA> rather than "nan"
            out[col] = out[col].replace({"nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "": pd.NA})

    # Convert timestamp/date-like columns to datetime (timezone-naive)
    for col in out.columns:
        if _is_datetime_like(out[col], col):
            dt = pd.to_datetime(out[col], errors="coerce")
            # Remove timezone if present
            if isinstance(dt.dtype, pd.DatetimeTZDtype):
                dt = dt.dt.tz_localize(None)
            out[col] = dt

    # Required keys: drop rows if these are missing (table-specific)
    required_keys: dict[str, list[str]] = {
        "lead_logs": ["id", "lead_id"],
        # Keep all referral rows for final output row-count expectations.
        # Missing referee/referrer can happen in source; handle downstream with defaults.
        "user_referrals": ["referral_id"],
        "user_referral_logs": ["id", "user_referral_id"],
        "user_logs": ["id", "user_id"],
        "user_referral_statuses": ["id"],
        "referral_rewards": ["id"],
        "paid_transactions": ["transaction_id"],
    }
    keys = [k for k in required_keys.get(table_name, []) if k in out.columns]
    if keys:
        out = out.dropna(subset=keys)

    # Reward-value rule: only replace nulls with 0 where logically correct.
    if table_name == "referral_rewards" and "reward_value" in out.columns:
        out["reward_value"] = pd.to_numeric(out["reward_value"], errors="coerce").fillna(0)

    # Enforce no nulls:
    # - numeric -> 0
    # - datetime -> default epoch-ish
    # - bool-like -> False
    # - strings -> empty string
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[col] = s.fillna(_DEFAULT_DATETIME)
        elif pd.api.types.is_bool_dtype(s):
            out[col] = s.fillna(False)
        elif pd.api.types.is_numeric_dtype(s):
            out[col] = s.fillna(0)
        else:
            # keep as string for consistency in outputs
            out[col] = s.astype("string").fillna("")

    # Final guarantee: no nulls anywhere
    null_total = int(out.isna().sum().sum())
    if null_total != 0:
        raise ValueError(f"{table_name}: expected no nulls after cleaning, found {null_total}")

    return out


def profile_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Column-level profiling for a single table:
    - column name
    - dtype
    - null count
    - distinct count (non-null)
    - min/max (numeric & dates)
    """
    rows: list[dict[str, object]] = []

    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)
        null_count = int(s.isna().sum())
        distinct_count = int(s.nunique(dropna=True))

        min_val: object | None = None
        max_val: object | None = None

        if pd.api.types.is_numeric_dtype(s):
            # Avoid warnings on all-null numeric columns
            if s.dropna().empty:
                min_val = None
                max_val = None
            else:
                min_val = s.min()
                max_val = s.max()
        elif _is_datetime_like(s, col):
            dt = pd.to_datetime(s, errors="coerce")
            if dt.dropna().empty:
                min_val = None
                max_val = None
            else:
                min_val = _excel_safe_datetime(dt.min())
                max_val = _excel_safe_datetime(dt.max())

        rows.append(
            {
                "table": table_name,
                "column": col,
                "dtype": dtype_str,
                "null_count": null_count,
                "distinct_count": distinct_count,
                "min": min_val,
                "max": max_val,
            }
        )

    return pd.DataFrame(rows)


def convert_utc_to_local(
    df: pd.DataFrame, timestamp_cols: list[str], timezone_col: str | None
) -> pd.DataFrame:
    """
    Convert UTC timestamps to local time using timezone column.
    If timezone_col is None, assumes UTC (no conversion needed).
    """
    out = df.copy()

    if timezone_col is None or timezone_col not in out.columns:
        # No timezone info available - assume already in desired timezone or UTC
        return out

    for ts_col in timestamp_cols:
        if ts_col not in out.columns:
            continue

        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(out[ts_col]):
            continue

        # Localize to UTC (assume timestamps start in UTC, timezone-naive)
        ts_utc = out[ts_col].dt.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward")

        # Vectorized approach: group by timezone and convert
        result = pd.Series(index=out.index, dtype="datetime64[ns]")

        for tz_name in out[timezone_col].dropna().unique():
            mask = out[timezone_col] == tz_name
            if not mask.any():
                continue

            try:
                tz = pytz.timezone(str(tz_name))
                ts_subset = ts_utc[mask]
                converted = ts_subset.dt.tz_convert(tz).dt.tz_localize(None)
                result[mask] = converted
            except (pytz.UnknownTimeZoneError, Exception):
                # Fallback: keep UTC if timezone is invalid
                converted = ts_utc[mask].dt.tz_localize(None)
                result[mask] = converted

        # Fill any remaining NaT with UTC-converted (fallback)
        result = result.fillna(ts_utc.dt.tz_localize(None))
        out[ts_col] = result

    return out


def apply_timezone_adjustments(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Step 5: Convert UTC timestamps to local time using timezone columns.
    If timezone column missing, join to another table that has timezone info.
    """
    result = {}

    # Tables with direct timezone columns
    # lead_logs: timezone_location -> created_at
    if "lead_logs" in tables:
        df = tables["lead_logs"]
        timestamp_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        result["lead_logs"] = convert_utc_to_local(df, timestamp_cols, "timezone_location")

    # user_logs: timezone_homeclub -> membership_expired_date
    if "user_logs" in tables:
        df = tables["user_logs"]
        timestamp_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        result["user_logs"] = convert_utc_to_local(df, timestamp_cols, "timezone_homeclub")

    # paid_transactions: timezone_transaction -> transaction_at
    if "paid_transactions" in tables:
        df = tables["paid_transactions"]
        timestamp_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        result["paid_transactions"] = convert_utc_to_local(df, timestamp_cols, "timezone_transaction")

    # user_referrals: join to user_logs via referrer_id to get timezone_homeclub
    if "user_referrals" in tables and "user_logs" in result:
        df = tables["user_referrals"]
        user_logs = result["user_logs"]
        # Create lookup dictionaries for timezones
        referrer_tz = dict(zip(user_logs["user_id"], user_logs["timezone_homeclub"]))
        referee_tz = dict(zip(user_logs["user_id"], user_logs["timezone_homeclub"]))
        
        # Map timezones
        merged = df.copy()
        merged["timezone_referrer"] = merged["referrer_id"].map(referrer_tz)
        merged["timezone_referee"] = merged["referee_id"].map(referee_tz)
        # Use referrer's timezone, fallback to referee's
        merged["timezone"] = merged["timezone_referrer"].fillna(merged["timezone_referee"])
        
        timestamp_cols = [c for c in merged.columns if pd.api.types.is_datetime64_any_dtype(merged[c]) and c not in ["timezone_referrer", "timezone_referee"]]
        result["user_referrals"] = convert_utc_to_local(merged, timestamp_cols, "timezone")
        # Drop helper columns
        result["user_referrals"] = result["user_referrals"].drop(
            columns=["timezone_referrer", "timezone_referee", "timezone"], errors="ignore"
        )

    # user_referral_logs: join to user_referrals, then to user_logs
    if "user_referral_logs" in tables and "user_referrals" in result and "user_logs" in result:
        df = tables["user_referral_logs"]
        user_referrals_orig = tables["user_referrals"]  # Use original cleaned table for join keys
        user_logs = result["user_logs"]
        # Create lookup: referral_id -> referrer_id -> timezone
        referral_to_referrer = dict(zip(user_referrals_orig["referral_id"], user_referrals_orig["referrer_id"]))
        referrer_to_tz = dict(zip(user_logs["user_id"], user_logs["timezone_homeclub"]))
        
        # Map timezone through referral -> referrer -> timezone
        merged = df.copy()
        merged["referrer_id_lookup"] = merged["user_referral_id"].map(referral_to_referrer)
        merged["timezone_homeclub"] = merged["referrer_id_lookup"].map(referrer_to_tz)
        
        timestamp_cols = [c for c in merged.columns if pd.api.types.is_datetime64_any_dtype(merged[c]) and c not in ["timezone_homeclub"]]
        result["user_referral_logs"] = convert_utc_to_local(merged, timestamp_cols, "timezone_homeclub")
        result["user_referral_logs"] = result["user_referral_logs"].drop(
            columns=["referrer_id_lookup", "timezone_homeclub"], errors="ignore"
        )

    # user_referral_statuses: no clear join path, skip timezone conversion (keep UTC)
    if "user_referral_statuses" in tables:
        result["user_referral_statuses"] = tables["user_referral_statuses"]

    # referral_rewards: no clear join path, skip timezone conversion (keep UTC)
    if "referral_rewards" in tables:
        result["referral_rewards"] = tables["referral_rewards"]

    return result


def build_final_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Step 6: Join tables in recommended order to build final dataset.
    After each join: check row count, remove duplicates, validate keys.
    """
    print("\n=== Building Final Dataset ===")

    # 1. Start with user_referrals (base table)
    result = tables["user_referrals"].copy()
    print(f"1. Base: user_referrals -> {len(result)} rows")

    # 2. Join user_referral_logs (one-to-many: one referral can have multiple logs)
    if "user_referral_logs" in tables:
        # Deduplicate logs to keep one row per referral (latest log wins)
        logs = tables["user_referral_logs"].copy()
        if "created_at" in logs.columns:
            logs = logs.sort_values("created_at", ascending=False)
        logs = logs.drop_duplicates(subset=["user_referral_id"], keep="first")
        before_count = len(result)
        result = result.merge(
            logs,
            left_on="referral_id",
            right_on="user_referral_id",
            how="left",
            suffixes=("", "_log"),
        )
        # Remove duplicates if any (shouldn't happen with proper keys, but check)
        result = result.drop_duplicates()
        print(f"2. Join user_referral_logs -> {len(result)} rows (was {before_count}, added {len(result) - before_count})")

    # 3. Join user_referral_statuses
    if "user_referral_statuses" in tables:
        before_count = len(result)
        result = result.merge(
            tables["user_referral_statuses"],
            left_on="user_referral_status_id",
            right_on="id",
            how="left",
            suffixes=("", "_status"),
        )
        result = result.drop_duplicates()
        print(f"3. Join user_referral_statuses -> {len(result)} rows (was {before_count})")

    # 4. Join referral_rewards
    if "referral_rewards" in tables:
        before_count = len(result)
        result = result.merge(
            tables["referral_rewards"],
            left_on="referral_reward_id",
            right_on="id",
            how="left",
            suffixes=("", "_reward"),
        )
        result = result.drop_duplicates()
        print(f"4. Join referral_rewards -> {len(result)} rows (was {before_count})")

    # 5. Join paid_transactions
    if "paid_transactions" in tables:
        before_count = len(result)
        result = result.merge(
            tables["paid_transactions"],
            left_on="transaction_id",
            right_on="transaction_id",
            how="left",
            suffixes=("", "_transaction"),
        )
        result = result.drop_duplicates()
        print(f"5. Join paid_transactions -> {len(result)} rows (was {before_count})")

    # 6. Join user_logs twice: once for referrer, once for referee
    # First, deduplicate user_logs (keep most recent per user_id based on membership_expired_date)
    if "user_logs" in tables:
        user_logs_dedup = tables["user_logs"].copy()
        # Sort by membership_expired_date descending (most recent first), then drop duplicates keeping first
        if "membership_expired_date" in user_logs_dedup.columns:
            user_logs_dedup = user_logs_dedup.sort_values("membership_expired_date", ascending=False)
        user_logs_dedup = user_logs_dedup.drop_duplicates(subset=["user_id"], keep="first")
        print(f"   -> Deduplicated user_logs: {len(tables['user_logs'])} -> {len(user_logs_dedup)} rows")

        before_count = len(result)
        # Join for referrer (rename all columns except user_id to avoid conflicts)
        referrer_cols = {
            col: f"{col}_referrer" if col != "user_id" else col
            for col in user_logs_dedup.columns
        }
        referrer_df = user_logs_dedup.rename(columns=referrer_cols)
        result = result.merge(
            referrer_df,
            left_on="referrer_id",
            right_on="user_id",
            how="left",
            suffixes=("", "_ref_skip"),
        )
        # Drop the user_id column from referrer join (we don't need it)
        if "user_id" in result.columns:
            result = result.drop(columns=["user_id"], errors="ignore")
        result = result.drop_duplicates()
        print(f"6a. Join user_logs (referrer) -> {len(result)} rows (was {before_count})")

        # Join for referee (rename all columns to avoid conflicts with referrer columns)
        referee_cols = {col: f"{col}_referee" for col in user_logs_dedup.columns}
        referee_df = user_logs_dedup.rename(columns=referee_cols)
        before_count = len(result)
        result = result.merge(
            referee_df,
            left_on="referee_id",
            right_on="user_id_referee",
            how="left",
            suffixes=("", "_referee_skip"),
        )
        # Drop the user_id_referee column
        if "user_id_referee" in result.columns:
            result = result.drop(columns=["user_id_referee"], errors="ignore")
        result = result.drop_duplicates()
        print(f"6b. Join user_logs (referee) -> {len(result)} rows (was {before_count})")

    # 7. Join lead_logs (for Lead source) - join on referee_id = lead_id
    if "lead_logs" in tables:
        # Deduplicate lead_logs (keep most recent per lead_id)
        lead_logs_dedup = tables["lead_logs"].copy()
        if "created_at" in lead_logs_dedup.columns:
            lead_logs_dedup = lead_logs_dedup.sort_values("created_at", ascending=False)
        lead_logs_dedup = lead_logs_dedup.drop_duplicates(subset=["lead_id"], keep="first")
        print(f"   -> Deduplicated lead_logs: {len(tables['lead_logs'])} -> {len(lead_logs_dedup)} rows")

        before_count = len(result)
        result = result.merge(
            lead_logs_dedup,
            left_on="referee_id",
            right_on="lead_id",
            how="left",
            suffixes=("", "_lead"),
        )
        result = result.drop_duplicates()
        print(f"7. Join lead_logs -> {len(result)} rows (was {before_count})")

    # Clean up duplicate ID columns from joins (keep original, drop suffixed versions)
    id_cols_to_drop = [col for col in result.columns if col.endswith(("_status", "_reward", "_transaction", "_lead", "_log")) and col.startswith("id")]
    if id_cols_to_drop:
        result = result.drop(columns=id_cols_to_drop, errors="ignore")
        print(f"   -> Dropped duplicate ID columns: {id_cols_to_drop}")

    # Final validation: check for duplicate referral_ids
    # Note: Some duplicates are expected from one-to-many joins (e.g., user_referral_logs)
    if "referral_id" in result.columns:
        dup_count = result["referral_id"].duplicated().sum()
        if dup_count > 0:
            print(f"INFO: Found {dup_count} duplicate referral_ids (expected from one-to-many joins like logs)")
            print(f"   -> Keeping all rows to preserve log information")

    print(f"\nFinal dataset: {len(result)} rows, {len(result.columns)} columns")
    return result


def finalize_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 10: Final output
    - Select only required columns in the correct order
    - Confirm: row count = 46, no nulls, dtypes make sense
    """
    required_columns = [
        "referral_id",
        "referral_at",
        "referral_source",
        "referral_source_category",
        "user_referral_status_id",
        "description",
        "referral_reward_id",
        "reward_type",
        "reward_value",
        "is_reward_granted",
        "transaction_id",
        "transaction_status",
        "transaction_type",
        "transaction_at",
        "transaction_location",
        "referrer_id",
        "name_referrer",
        "phone_number_referrer",
        "homeclub_referrer",
        "membership_expired_date_referrer",
        "is_deleted_referrer",
        "referee_id",
        "referee_name",
        "referee_phone",
        "homeclub_referee",
        "membership_expired_date_referee",
        "is_deleted_referee",
        "source_category",
        "preferred_location",
        "current_status",
        "is_business_logic_valid",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"finalize_output: missing required columns: {missing}")

    out = df.loc[:, required_columns].copy()

    # Confirm row count expectation
    if len(out) != 46:
        raise ValueError(f"finalize_output: expected 46 rows, got {len(out)}")

    # Enforce "no nulls" contract after left-joins:
    # - datetime -> _DEFAULT_DATETIME
    # - numeric -> 0
    # - bool -> False
    # - strings/ids -> "Unknown" (or "UNKNOWN" for club-like)
    club_cols = [c for c in out.columns if "club" in c.lower()]
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[col] = s.fillna(_DEFAULT_DATETIME)
        elif pd.api.types.is_bool_dtype(s):
            out[col] = s.fillna(False)
        elif pd.api.types.is_numeric_dtype(s):
            out[col] = s.fillna(0)
        else:
            fill_value = "UNKNOWN" if col in club_cols else "Unknown"
            out[col] = s.astype("string").fillna("")
            out.loc[out[col].str.strip() == "", col] = fill_value

    # Confirm no nulls
    null_total = int(out.isna().sum().sum())
    if null_total != 0:
        raise ValueError(f"finalize_output: expected no nulls, found {null_total}")

    # Light dtype sanity: boolean must be bool; numeric reward_value must be numeric-ish
    if out["is_business_logic_valid"].dtype != bool:
        out["is_business_logic_valid"] = out["is_business_logic_valid"].astype(bool)
    out["reward_value"] = pd.to_numeric(out["reward_value"], errors="coerce").fillna(0)

    return out


def validate_business_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 9: Implement business logic validation.
    Create is_business_logic_valid boolean column.
    
    Valid conditions (ALL must be True):
    1. Reward > 0
    2. Status = "Berhasil"
    3. Transaction exists
    4. Transaction is PAID
    5. Transaction type = NEW
    6. Transaction after referral
    7. Same month (transaction and referral)
    8. Referrer membership not expired
    9. Referrer not deleted
    10. Reward granted
    
    Default to False (invalid) unless explicitly valid.
    """
    out = df.copy()
    
    # Initialize to False (invalid by default)
    out["is_business_logic_valid"] = False

    def _norm_str(s: pd.Series) -> pd.Series:
        # Normalize for safe string comparisons; keep missing as <NA>.
        return s.astype("string").str.strip().str.lower()

    def _parse_bool_explicit(
        s: pd.Series,
        *,
        true_values: set[str],
        false_values: set[str],
        default: bool = False,
    ) -> pd.Series:
        """
        Parse a boolean-ish column with an explicit default.
        Any value not recognized as true/false becomes default.
        This matches "default invalid unless explicitly valid".
        """
        if pd.api.types.is_bool_dtype(s):
            # Missing values default.
            return s.fillna(default)

        normalized = _norm_str(s)
        is_true = normalized.isin(true_values)
        is_false = normalized.isin(false_values)
        # Unknown / missing -> default
        out_bool = pd.Series(default, index=s.index, dtype="boolean")
        out_bool = out_bool.mask(is_true, True)
        out_bool = out_bool.mask(is_false, False)
        return out_bool.fillna(default)
    
    # Rule 1: Reward > 0
    if "reward_value" in out.columns:
        rule1 = (pd.to_numeric(out["reward_value"], errors="coerce") > 0).fillna(False)
    else:
        rule1 = pd.Series(False, index=out.index)
    
    # Rule 2: Status = "Berhasil"
    if "description" in out.columns:
        rule2 = (_norm_str(out["description"]) == "berhasil").fillna(False)
    else:
        rule2 = pd.Series(False, index=out.index)
    
    # Rule 3: Transaction exists (transaction_id is not null/empty)
    if "transaction_id" in out.columns:
        tx_id = out["transaction_id"].astype("string")
        rule3 = (tx_id.notna() & (tx_id.str.strip() != "")).fillna(False)
    else:
        rule3 = pd.Series(False, index=out.index)
    
    # Rule 4: Transaction is PAID
    if "transaction_status" in out.columns:
        rule4 = (_norm_str(out["transaction_status"]) == "paid").fillna(False)
    else:
        rule4 = pd.Series(False, index=out.index)
    
    # Rule 5: Transaction type = NEW
    if "transaction_type" in out.columns:
        rule5 = (_norm_str(out["transaction_type"]) == "new").fillna(False)
    else:
        rule5 = pd.Series(False, index=out.index)
    
    # Rule 6: Transaction after referral (transaction_at > referral_at)
    if "transaction_at" in out.columns and "referral_at" in out.columns:
        # Ensure datetime types
        trans_at = pd.to_datetime(out["transaction_at"], errors="coerce")
        ref_at = pd.to_datetime(out["referral_at"], errors="coerce")
        rule6 = (trans_at.notna() & ref_at.notna() & (trans_at > ref_at)).fillna(False)
    else:
        rule6 = pd.Series(False, index=out.index)
    
    # Rule 7: Same month (transaction and referral in same month)
    if "transaction_at" in out.columns and "referral_at" in out.columns:
        trans_at = pd.to_datetime(out["transaction_at"], errors="coerce")
        ref_at = pd.to_datetime(out["referral_at"], errors="coerce")
        rule7 = (
            trans_at.notna() & 
            ref_at.notna() & 
            (trans_at.dt.year == ref_at.dt.year) & 
            (trans_at.dt.month == ref_at.dt.month)
        ).fillna(False)
    else:
        rule7 = pd.Series(False, index=out.index)
    
    # Rule 8: Referrer membership not expired
    # Membership is valid if expired_date is after transaction_at (or referral_at if no transaction)
    if "membership_expired_date_referrer" in out.columns:
        expired_date = pd.to_datetime(out["membership_expired_date_referrer"], errors="coerce")
        # Use transaction_at if available, otherwise referral_at
        if "transaction_at" in out.columns:
            check_date = pd.to_datetime(out["transaction_at"], errors="coerce")
        elif "referral_at" in out.columns:
            check_date = pd.to_datetime(out["referral_at"], errors="coerce")
        else:
            check_date = pd.Series(pd.NaT, index=out.index)
        
        rule8 = (expired_date.notna() & check_date.notna() & (expired_date > check_date)).fillna(False)
    else:
        rule8 = pd.Series(False, index=out.index)
    
    # Rule 9: Referrer not deleted
    if "is_deleted_referrer" in out.columns:
        # Explicit: only accept known "not deleted" values as True.
        # Unknown/missing -> invalid (False) to respect default-invalid policy.
        deleted = out["is_deleted_referrer"]
        is_deleted = _parse_bool_explicit(
            deleted,
            true_values={"true", "1", "yes", "y"},
            false_values={"false", "0", "no", "n"},
            default=False,
        )
        rule9 = (~is_deleted).fillna(False)
    else:
        rule9 = pd.Series(False, index=out.index)
    
    # Rule 10: Reward granted
    if "is_reward_granted" in out.columns:
        # Explicit: only accept known "granted" values as True.
        # Unknown/missing -> invalid (False).
        granted = out["is_reward_granted"]
        rule10 = _parse_bool_explicit(
            granted,
            true_values={"true", "1", "yes", "y"},
            false_values={"false", "0", "no", "n"},
            default=False,
        ).fillna(False)
    else:
        rule10 = pd.Series(False, index=out.index)
    
    # All rules must be True for validation to pass
    out["is_business_logic_valid"] = (
        rule1 & rule2 & rule3 & rule4 & rule5 & 
        rule6 & rule7 & rule8 & rule9 & rule10
    ).fillna(False).astype(bool)
    
    return out


def derive_referral_source_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 8: Derive referral_source_category using CASE logic:
    - "User Sign Up" → "Online"
    - "Draft Transaction" → "Offline"
    - Lead (when source_category exists) → leads.source_category
    """
    out = df.copy()

    # Initialize the new column
    out["referral_source_category"] = ""

    # Case 1: User Sign Up → Online
    if "referral_source" in out.columns:
        mask_user_signup = out["referral_source"].astype("string").str.strip().str.lower() == "user sign up"
        out.loc[mask_user_signup, "referral_source_category"] = "Online"

        # Case 2: Draft Transaction → Offline
        mask_draft = out["referral_source"].astype("string").str.strip().str.lower() == "draft transaction"
        out.loc[mask_draft, "referral_source_category"] = "Offline"

        # Case 3: Lead → use source_category from lead_logs (if available)
        if "source_category" in out.columns:
            mask_lead = ~mask_user_signup & ~mask_draft & out["source_category"].notna()
            mask_lead = mask_lead & (out["source_category"].astype("string").str.strip() != "")
            out.loc[mask_lead, "referral_source_category"] = out.loc[mask_lead, "source_category"]

    return out


def _initcap(value: str) -> str:
    # Similar to SQL INITCAP: title-case words.
    return " ".join(part.capitalize() for part in value.split())


def _is_missing_string(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"<na>", "nan", "none"}


def apply_string_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 7: String formatting rules
    - InitCap for names and general strings
    - Do NOT change club names (keep uppercase / as-is)
    - Normalize transaction_status -> "Paid", transaction_type -> "New"
    """
    out = df.copy()

    # Columns we never "pretty format" (identifiers / codes / timezones / phones)
    no_format_substrings = (
        "_id",
        "timezone",
        "phone",
    )

    # Club names: keep as-is (they are expected uppercase in source)
    club_cols = [c for c in out.columns if "club" in c.lower()]

    # InitCap for general string columns
    for col in out.columns:
        if col in club_cols:
            continue
        if any(sub in col.lower() for sub in no_format_substrings):
            continue
        if not (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])):
            continue

        out[col] = out[col].astype("string").map(lambda x: "" if _is_missing_string(x) else _initcap(str(x)))

    # Normalize key enums (case-insensitive)
    if "transaction_status" in out.columns:
        out["transaction_status"] = out["transaction_status"].astype("string").map(
            lambda x: "" if _is_missing_string(x) else str(x).strip()
        )
        out["transaction_status"] = out["transaction_status"].replace(
            {r"(?i)^paid$": "Paid"}, regex=True
        )
        # If anything else remains non-empty, standardize to InitCap
        out["transaction_status"] = out["transaction_status"].map(lambda x: _initcap(str(x)) if str(x).strip() else "")

    if "transaction_type" in out.columns:
        out["transaction_type"] = out["transaction_type"].astype("string").map(
            lambda x: "" if _is_missing_string(x) else str(x).strip()
        )
        out["transaction_type"] = out["transaction_type"].replace(
            {r"(?i)^new$": "New"}, regex=True
        )
        out["transaction_type"] = out["transaction_type"].map(lambda x: _initcap(str(x)) if str(x).strip() else "")

    # Final pass: avoid blanks in human-facing string columns (keeps "no nulls" spirit in exports)
    for col in out.columns:
        if not (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])):
            continue
        if any(sub in col.lower() for sub in no_format_substrings):
            continue

        fill_value = "UNKNOWN" if col in club_cols else "Unknown"
        out[col] = out[col].astype("string").fillna("")
        out.loc[out[col].str.strip() == "", col] = fill_value

    return out


def write_profiling_report(tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for table_name, df in tables.items():
            prof = profile_table(df, table_name)
            # Excel sheet names have a 31-char limit.
            sheet_name = table_name[:31]
            prof.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    profiling_path = project_root / "profiling" / "data_profiling.xlsx"

    lead_logs_df = pd.read_csv(data_dir / "lead_logs.csv")
    user_referrals_df = pd.read_csv(data_dir / "user_referrals.csv")
    user_referral_logs_df = pd.read_csv(data_dir / "user_referral_logs.csv")
    user_logs_df = pd.read_csv(data_dir / "user_logs.csv")
    user_referral_statuses_df = pd.read_csv(data_dir / "user_referral_statuses.csv")
    referral_rewards_df = pd.read_csv(data_dir / "referral_rewards.csv")
    paid_transactions_df = pd.read_csv(data_dir / "paid_transactions.csv")

    tables: dict[str, pd.DataFrame] = {
        "lead_logs": lead_logs_df,
        "user_referrals": user_referrals_df,
        "user_referral_logs": user_referral_logs_df,
        "user_logs": user_logs_df,
        "user_referral_statuses": user_referral_statuses_df,
        "referral_rewards": referral_rewards_df,
        "paid_transactions": paid_transactions_df,
    }

    # Step 4: clean each table before any joins.
    cleaned_tables: dict[str, pd.DataFrame] = {name: clean_table(df, name) for name, df in tables.items()}

    # Step 5: timezone adjustment - convert UTC to local time
    timezone_adjusted_tables = apply_timezone_adjustments(cleaned_tables)

    # Basic sanity check: one print per table (shape + columns).
    for name, df in timezone_adjusted_tables.items():
        print(f"{name}: shape={df.shape}")
        print(f"{name}: columns={list(df.columns)}")
        print("-" * 80)

    write_profiling_report(timezone_adjusted_tables, profiling_path)
    print(f"Profiling report written to: {profiling_path}")

    # Step 6: Join tables to build final dataset
    final_dataset = build_final_dataset(timezone_adjusted_tables)

    # Step 8: Derive referral_source_category
    final_dataset = derive_referral_source_category(final_dataset)

    # Step 9: Implement business logic validation
    final_dataset = validate_business_logic(final_dataset)

    # Step 7: Apply presentation-friendly string formatting
    final_dataset = apply_string_formatting(final_dataset)

    # Step 10: Final output selection + validation
    final_dataset = finalize_output(final_dataset)
    
    # Save final dataset
    output_path = project_root / "output" / "referral_report.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_dataset.to_csv(output_path, index=False)
    print(f"\nFinal dataset saved to: {output_path}")


if __name__ == "__main__":
    main()

