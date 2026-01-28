import pandas as pd

df = pd.read_csv('output/referral_report.csv')
sample = df.iloc[4]  # Row with transaction data
print('Sample row checks:')
print(f'reward_value: {sample["reward_value"]} (>{0}? {float(sample["reward_value"]) > 0})')
print(f'description: "{sample["description"]}" (=Berhasil? {str(sample["description"]).strip().lower() == "berhasil"})')
print(f'transaction_status: "{sample["transaction_status"]}" (=Paid? {str(sample["transaction_status"]).strip().lower() == "paid"})')
print(f'transaction_type: "{sample["transaction_type"]}" (=New? {str(sample["transaction_type"]).strip().lower() == "new"})')
print(f'is_reward_granted: {sample["is_reward_granted"]}')
print(f'is_deleted_referrer: {sample["is_deleted_referrer"]}')
print(f'\nValidation result: {sample["is_business_logic_valid"]}')
