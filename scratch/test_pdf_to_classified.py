import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from classify_pipeline import classify_and_save_artifacts, get_classification_engine

def test_pipeline():
    print("Loading test dataframe...")
    test_df = pd.read_excel("verified_transactions.xlsx")
    print(f"Loaded {len(test_df)} rows. Columns: {list(test_df.columns)}")

    print("\nInitializing classification engine...")
    model, sub_embeddings = get_classification_engine()

    print("\nExecuting classify_and_save_artifacts...")
    classified_df, emb_path, excel_path = classify_and_save_artifacts(
        df=test_df,
        source_name="00007_bank_statement.pdf",
        output_base_dir="output",
        model=model,
        sub_embeddings=sub_embeddings,
        desc_col="Description"
    )

    print("\n--- RESULTS VERIFICATION ---")
    print(f"Embeddings saved to: {emb_path}")
    print(f"Excel saved to: {excel_path}")

    assert os.path.exists(emb_path), f"Embeddings file not found: {emb_path}"
    assert os.path.exists(excel_path), f"Excel file not found: {excel_path}"

    embs = np.load(emb_path)
    print(f"Embeddings array shape: {embs.shape}")
    assert embs.shape[0] == len(test_df)
    assert embs.shape[1] == 768

    print(f"\nClassified Columns: {list(classified_df.columns)}")
    for col in ["Transaction Mode", "Transaction Direction", "Transaction Purpose", "Bank / Institution", "Party / Counterparty"]:
        assert col in classified_df.columns, f"Missing required column: {col}"

    score_cols = [c for c in classified_df.columns if "score" in c.lower() or "sim" in c.lower()]
    print(f"Score columns found (should be empty): {score_cols}")
    assert len(score_cols) == 0, "Score columns must NOT be present!"

    print("\nSample Preview:")
    preview_cols = ["Description", "Debit", "Credit", "Transaction Mode", "Transaction Direction", "Transaction Purpose", "Bank / Institution", "Party / Counterparty"]
    print(classified_df[preview_cols].head(8).to_string(index=False))

    print("\nSUCCESS: All pipeline and storage assertions passed!")

if __name__ == "__main__":
    test_pipeline()
