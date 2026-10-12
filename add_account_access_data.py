
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset_path = (
    PROJECT_ROOT
    / "data"
    / "tickets"
    / "02_improved_support_tickets.csv"
)

df = pd.read_csv(dataset_path)

new_queries = [
    "I cannot sign in to my internet banking account",
    "My online banking password is not working",
    "My bank account login has been locked",
    "I forgot my internet banking username",
    "The banking app keeps rejecting my password",
    "I am unable to reset my login credentials",
    "My mobile banking account is blocked",
    "I cannot access my account after changing phones",
    "The login OTP is not arriving on my registered number",
    "My account access was disabled after multiple login attempts",
    "I receive an authentication error when signing in",
    "I cannot log in using my customer ID",
    "My internet banking session expires immediately",
    "The banking app logs me out automatically",
    "I am unable to activate internet banking",
    "My login credentials are showing as invalid",
    "I need help unlocking my online banking account",
    "I cannot access mobile banking after reinstalling the app",
    "My biometric login is not working",
    "I cannot register my device for mobile banking",
    "The banking website says my user ID is blocked",
    "I am unable to change my internet banking password",
    "The app freezes whenever I try to log in",
    "My two-factor authentication is failing",
    "I cannot access my account from a new device",
    "My banking login verification code has expired",
    "I am unable to enable fingerprint login"
]

new_data = pd.DataFrame({
    "query_text": new_queries,
    "category": "Account Access"
})

updated_df = pd.concat(
    [df, new_data],
    ignore_index=True
)

updated_df = updated_df.drop_duplicates(
    subset=["query_text", "category"]
)

updated_df.to_csv(dataset_path, index=False)

print("\nAccount Access Dataset Updated")
print("Total examples:", len(updated_df))

print("\nAccount Access examples:")
print(
    (updated_df["category"] == "Account Access").sum()
)

print("\nCategory Distribution:")
print(updated_df["category"].value_counts())


#Improve Account Services

