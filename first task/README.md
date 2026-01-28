# Referral Analytics Project

Clean project structure for referral analytics work.

## Layout

```
project/
│
├── data/
│   ├── lead_logs.csv
│   ├── user_referrals.csv
│   ├── user_referral_logs.csv
│   ├── user_logs.csv
│   ├── user_referral_statuses.csv
│   ├── referral_rewards.csv
│   └── paid_transactions.csv
│
├── output/
│   └── referral_report.csv
│
├── profiling/
│   └── data_profiling.xlsx
│
├── src/
│   └── main.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Notes

- `dataset/` (if present) is treated as raw input; canonical working copies are in `data/`.
- `output/` is for generated artifacts (reports, exports).
- `profiling/` is for exploratory profiling results.

## Run locally

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.main
```

