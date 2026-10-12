
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
    "My KYC verification is still pending",
    "The bank rejected my identity proof during KYC",
    "How can I complete video KYC from home?",
    "My PAN verification failed during account verification",
    "I uploaded my Aadhaar but KYC is not approved",
    "The bank is asking me to complete re-KYC",
    "My KYC documents are under review for several days",
    "I received a notification that my KYC has expired",
    "How can I check the status of my KYC application?",
    "My address proof was rejected during KYC verification",
    "I cannot complete biometric verification for KYC",
    "What documents are accepted for customer identity verification?",
    "My video KYC appointment was cancelled",
    "The bank has restricted my account due to incomplete KYC",
    "I need to update my PAN details for KYC compliance",
    "My photograph was rejected during online KYC",
    "The KYC portal is not accepting my identity documents",
    "I completed KYC but the bank still shows verification pending",
    "How do I submit documents for periodic KYC renewal?",
    "My date of birth does not match the KYC records",
    "My passport details are not updating in the bank records",
    "I cannot update my passport information in my bank KYC",
    "My passport number is incorrect in the bank records",
    "The bank has not updated my renewed passport details",
    "How can I correct my passport information in KYC?",
    "My identity document update is pending with the bank",
    "The bank rejected my passport during KYC verification",
    "My KYC documents are not reflecting in my account",
    "My updated address proof is not showing in KYC records",
    "My name on the passport does not match my bank KYC details",
    "I submitted my passport for KYC but the update is still pending",
    "How can I update my identification documents with the bank?"
]

new_data = pd.DataFrame({
    "query_text": new_queries,
    "category": "KYC"
})

updated_df = pd.concat(
    [df, new_data],
    ignore_index=True
)

updated_df = updated_df.drop_duplicates(
    subset=["query_text", "category"]
)

updated_df.to_csv(dataset_path, index=False)

print("\nKYC Dataset Updated")
print("Total examples:", len(updated_df))

print("\nKYC examples:")
print(
    (updated_df["category"] == "KYC").sum()
)

print("\nCategory Distribution:")
print(updated_df["category"].value_counts())
