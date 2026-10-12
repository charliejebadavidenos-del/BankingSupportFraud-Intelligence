
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
    "My UPI payment failed but money was debited",
    "I transferred money through NEFT but the beneficiary has not received it",
    "My IMPS transaction is showing pending",
    "The merchant did not receive my payment",
    "My electricity bill payment was unsuccessful",
    "I was charged twice for the same online purchase",
    "My bank transfer has not been processed",
    "The amount was deducted but the ATM did not dispense cash",
    "My cheque deposit has not been cleared",
    "The recipient says my transfer has not arrived",
    "My UPI transaction is stuck in processing",
    "I sent money to the wrong beneficiary by mistake",
    "My scheduled bank transfer did not execute",
    "Why was my international remittance delayed?",
    "My debit card payment was declined at the shop",
    "The payment gateway timed out during my transaction",
    "My credit card bill payment is not reflected",
    "The bank has not credited my inward remittance",
    "My cheque clearance is taking longer than expected",
    "My refund has not reached my bank account",
    "I made a payment but received no transaction confirmation",
    "My salary transfer is still pending",
    "My UPI collect request payment is failing",
    "I cannot complete an RTGS transfer",
    "My standing instruction payment was not processed",
    "The money transfer shows successful but the receiver has no credit",
    "My bank transfer was reversed without explanation"
]

new_data = pd.DataFrame({
    "query_text": new_queries,
    "category": "Transaction/Payment Issue"
})

updated_df = pd.concat(
    [df, new_data],
    ignore_index=True
)

updated_df = updated_df.drop_duplicates(
    subset=["query_text", "category"]
)

updated_df.to_csv(dataset_path, index=False)

print("\nTransaction/Payment Dataset Updated")
print("Total examples:", len(updated_df))

print("\nTransaction/Payment Issue examples:")
print(
    (updated_df["category"] == "Transaction/Payment Issue").sum()
)

print("\nCategory Distribution:")
print(updated_df["category"].value_counts())
