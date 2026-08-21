import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def load_data(file_path="extracted.xlsx"):
    """Loads the Excel file (checks for extracted.xlsx or fallback to extracted_full.xlsx)."""
    if not os.path.exists(file_path):
        fallback = "extracted_full.xlsx"
        if os.path.exists(fallback):
            print(f"'{file_path}' not found. Using '{fallback}' instead.")
            file_path = fallback
        else:
            raise FileNotFoundError(f"Neither '{file_path}' nor '{fallback}' was found.")
    
    print(f"Loading data from: {file_path}")
    df = pd.read_excel(file_path)
    return df, file_path

def resolve_target_column(df, target_col):
    """
    Finds the target column whether given as:
    1. An exact column name in df.columns
    2. An integer index (e.g. 0, 3) or string digit ('3')
    3. A text value found inside the table's header rows (e.g. 'Description', 'Narration', 'Txn Date')
    """
    if target_col in df.columns:
        return target_col
    
    cols_str = {str(c).lower().strip(): c for c in df.columns}
    target_str = str(target_col).lower().strip()
    if target_str in cols_str:
        return cols_str[target_str]

    if isinstance(target_col, int) and target_col < len(df.columns):
        return df.columns[target_col]
    if isinstance(target_col, str) and target_col.isdigit():
        idx = int(target_col)
        if idx < len(df.columns):
            return df.columns[idx]

    for col in df.columns:
        matching = df[df[col].astype(str).str.lower().str.strip() == target_str]
        if not matching.empty:
            print(f"Found header '{target_col}' inside table data -> Mapped to Column {col}")
            return col

    for col in df.columns:
        matching = df[df[col].astype(str).str.lower().str.contains(target_str, na=False)]
        if not matching.empty:
            print(f"Matched '{target_col}' in Column {col}")
            return col

    raise ValueError(
        f"Could not find column '{target_col}'. Available columns in Excel: {list(df.columns)}"
    )

def generate_and_save_embeddings(
    input_file="extracted.xlsx",
    target_column="Description",
    output_npy="embeddings.npy",
    model_name="nomic-ai/nomic-embed-text-v1.5"
):
    """
    Reads the Excel file, extracts text from the target column, generates
    Nomic Embeddings locally using SentenceTransformers, and saves them to a .npy file.
    """
    # 1. Load Excel Data
    df, resolved_path = load_data(input_file)
    resolved_col = resolve_target_column(df, target_column)
    
    text_data = df[resolved_col].astype(str).fillna('').tolist()
    print(f"Total rows to embed from column '{resolved_col}': {len(text_data)}")

    # 2. Load open-source Nomic model locally (No API key required)
    print(f"\nLoading open-source embedding model: {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # 3. Add Nomic's required prefix
    prefixed_texts = [f"search_document: {t}" if t.strip() else "search_document: empty" for t in text_data]

    # 4. Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(prefixed_texts, show_progress_bar=True)
    embeddings = np.array(embeddings)

    # 5. Save to .npy file
    np.save(output_npy, embeddings)
    print(f"\nSuccessfully saved embeddings to: '{output_npy}'")
    print(f"Saved Embedding Array Shape: {embeddings.shape}")
    print(f"Sample values (row 0, first 5 dimensions): {embeddings[0][:5]}")

    return embeddings

def main():
    # -------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------
    input_file = "extracted.xlsx"       # Path to Excel file
    target_column = "Description"       # Column name or index to embed
    output_npy = "embeddings.npy"       # Output .npy file path
    # -------------------------------------------------------------

    generate_and_save_embeddings(
        input_file=input_file,
        target_column=target_column,
        output_npy=output_npy
    )

if __name__ == "__main__":
    main()
