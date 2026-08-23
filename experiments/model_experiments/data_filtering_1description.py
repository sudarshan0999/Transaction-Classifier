import os
import re
import pandas as pd

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

def resolve_target_column(df, target_col="Description"):
    """Finds target column by name, index, or header row search."""
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

    raise ValueError(f"Could not find column '{target_col}'. Available columns: {list(df.columns)}")

def clean_description(raw_text):
    """
    Cleans raw bank transaction narrative by removing:
    - Reference IDs, UTR numbers, and numeric tokens (e.g. CNRBR946608852502, 714240285711, 414806262214)
    - IFSC Codes (e.g. HDFC0004171, ICIC0SF0002, ICIC0000011)
    - Phone / Account / ATM numbers (e.g. 7221215049, 96012)
    - Noise tags (/NONE, /URGENT/, SENT-TRANSFER FROM, UPI-TRANSFER TO)
    Preserves payment modes (UPI, NEFT, RTGS, IMPS, CASH, ATM, Cheque Clearing, etc.) and party/purpose details.
    """
    if pd.isna(raw_text) or not str(raw_text).strip():
        return ""
    
    text = str(raw_text).strip()
    
    # 1. Merge words split by Camelot across line breaks (e.g., RAMES\nH -> RAMESH)
    text = re.sub(r'\b(RAMES|MANIS|RAJES|HEALTHC|INVESTME|WELLNES)\s*\n\s*([A-Z]+)\b', r'\1\2', text, flags=re.IGNORECASE)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)

    # 2. Remove noise tags and transfer boilerplate with phone/account numbers
    text = re.sub(r'/NONE\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'/URGENT/?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'SENT-TRANSFER\s+FROM\s+\d*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'UPI-TRANSFER\s+TO\s+\d*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'FROM\s+\d{6,}', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'TO\s+\d{6,}', ' ', text, flags=re.IGNORECASE)

    # 3. Remove bank IFSC codes (e.g. HDFC0004171, ICIC0SF0002, ICIC0000011)
    text = re.sub(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', ' ', text, flags=re.IGNORECASE)

    # 4. Remove all numeric tokens / reference IDs containing digits (e.g. CNRBR946608852502, 714240285711, 96012)
    text = re.sub(r'\b[A-Za-z0-9]*\d+[A-Za-z0-9]*\b', ' ', text)

    # 5. Clean punctuation / redundant separators while preserving payment mode hyphens
    text = re.sub(r'[-–—]+', '-', text)
    text = re.sub(r'[/\\|_]+', ' ', text)
    text = re.sub(r'(?:\s*-\s*)+', ' - ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' -:,./')

    return text

def filter_dataset(df, target_col="Description"):
    """
    Cleans the target column, promotes valid headers if embedded in raw extraction,
    and removes non-transaction header/metadata rows.
    """
    resolved_col = resolve_target_column(df, target_col)
    
    # 1. If headers are embedded in table rows, locate and promote them
    header_idx = None
    for idx, row in df.iterrows():
        row_str = ' '.join(row.dropna().astype(str).str.lower())
        if 'description' in row_str and ('txn date' in row_str or 'date' in row_str or 'cheque' in row_str):
            header_idx = idx
            break
            
    if header_idx is not None and not str(df.columns[0]).lower().startswith('txn'):
        headers = [re.sub(r'\s+', ' ', str(h).strip().replace('\n', ' ')) for h in df.iloc[header_idx]]
        clean_df = df.iloc[header_idx + 1:].copy()
        clean_df.columns = headers
        resolved_col = 'Description'
    else:
        clean_df = df.copy()

    # 2. Filter out empty description rows & repetitive table headers
    is_invalid = (
        clean_df[resolved_col].isna() |
        clean_df[resolved_col].astype(str).str.strip().eq('') |
        clean_df[resolved_col].astype(str).str.lower().str.strip().isin(['description', 'narration', 'particulars', 'nan']) |
        clean_df[clean_df.columns[0]].astype(str).str.lower().str.contains('txn date|date|account holders|customer id|total', regex=True, na=False)
    )
    
    clean_df = clean_df[~is_invalid].copy().reset_index(drop=True)
    print(f"Filtered out {is_invalid.sum()} non-transaction / header rows. Remaining valid transactions: {len(clean_df)}")

    # 3. Clean the description and store back in 'Description'
    clean_df['Description'] = clean_df[resolved_col].apply(clean_description)

    # 4. Clean extra newlines & watermark text across all string columns
    for col in clean_df.columns:
        if col != 'Description':
            clean_df[col] = (
                clean_df[col]
                .astype(str)
                .replace(r'nan|None', '', regex=True)
                .str.replace('SYNTHETIC DATA - MACHINE GENERATED', '', regex=False)
                .str.replace('\n', ' ', regex=False)
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
            )

    # 5. Filter out any remaining blank descriptions
    clean_df = clean_df[clean_df['Description'] != ''].reset_index(drop=True)

    return clean_df

def save_data(df, output_excel, output_csv=None):
    """Saves DataFrame to Excel and CSV, handling file permission locks gracefully."""
    print(f"\nSaving cleaned dataset...")
    try:
        df.to_excel(output_excel, index=False)
        print(f"  - Excel: {output_excel}")
    except PermissionError:
        alt_excel = os.path.splitext(output_excel)[0] + "_cleaned.xlsx"
        print(f"  [Notice] '{output_excel}' is currently open in Excel. Saving to '{alt_excel}' instead.")
        df.to_excel(alt_excel, index=False)
        print(f"  - Excel (Alternative): {alt_excel}")

    if output_csv:
        try:
            df.to_csv(output_csv, index=False)
            print(f"  - CSV:   {output_csv}")
        except PermissionError:
            alt_csv = os.path.splitext(output_csv)[0] + "_cleaned.csv"
            df.to_csv(alt_csv, index=False)
            print(f"  - CSV (Alternative): {alt_csv}")

def main():
    input_file = "extracted.xlsx"
    target_column = "Description"
    output_excel = "filtered_description.xlsx"
    output_csv = "filtered_description.csv"

    # 1. Load Data
    df, resolved_path = load_data(input_file)
    print(f"Initial shape: {df.shape}")

    # 2. Clean & Filter Data
    filtered_df = filter_dataset(df, target_col=target_column)

    # 3. Display Sample Output
    print("\n" + "=" * 80)
    print(f"{'INDEX':<6} | {'CLEANED MEANINGFUL DESCRIPTION'}")
    print("=" * 80)
    for idx, row in filtered_df.head(20).iterrows():
        print(f"{idx:<6} | {row['Description']}")
    print("=" * 80)

    # 4. Save Cleaned Dataset back to Excel & CSV
    save_data(filtered_df, output_excel=output_excel, output_csv=output_csv)

if __name__ == "__main__":
    main()
