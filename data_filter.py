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

def extract_channel(raw_text):
    """Detects transaction channel (UPI, NEFT, RTGS, IMPS, Cash, Cheque, Clearing)."""
    if pd.isna(raw_text):
        return "Unknown"
    text_upper = str(raw_text).upper()
    if "UPI" in text_upper:
        return "UPI"
    elif "RTGS" in text_upper:
        return "RTGS"
    elif "NEFT" in text_upper:
        return "NEFT"
    elif "IMPS" in text_upper:
        return "IMPS"
    elif any(k in text_upper for k in ["CASH-BNA", "ATM WDL", "CASH WITHDRAWAL", "ATM CASH", "CASH"]):
        return "Cash / ATM"
    elif any(k in text_upper for k in ["CHQ", "CHEQUE", "CLEARING", "BY CLG"]):
        return "Cheque / Clearing"
    return "Bank Transfer"

def clean_description(raw_text):
    """
    Cleans raw bank transaction narrative by removing:
    - Reference IDs & UTR numbers (e.g. CNRBR946608852502, 714240285711, 414806262214)
    - IFSC Codes (e.g. HDFC0004171, ICIC0SF0002, ICIC0000011)
    - Phone / Account / ATM numbers (e.g. 7221215049, 96012)
    - Boilerplate tags (/NONE, --, /URGENT/, SENT-TRANSFER FROM, MICR Inward Clearing, etc.)
    - Extra punctuation, hyphens, and newlines
    """
    if pd.isna(raw_text) or not str(raw_text).strip():
        return ""
    
    text = str(raw_text).strip()
    
    # 1. Normalize linebreaks and multi-spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)

    # 2. Remove common boilerplate banking prefixes/tags
    boilerplate_patterns = [
        r'/NONE',
        r'/URGENT/?',
        r'--+',
        r'SENT-TRANSFER\s+FROM\s+\d+',
        r'TRANSFER\s+FROM\s+\d+',
        r'FROM\s+\d{6,}',
        r'By\s+Clg\s*:\s*DEL\s+ACCTS\s*-\s*CITI\s+BANK\s+N\.A\.\(CIT\)\s*,?',
        r'Chq\s+Paid\s*-\s*MICR\s+Inward\s+Clearing\s*-\s*',
        r'HDFC\s+BANK\s+LTD\.?',
        r'CITI\s+BANK\s+N\.A\.\(CIT\)',
        r'IMPS\s+BRN\s+SALARY\s+TRF\s+BY\s*-\s*',
        r'FAILED\s*-\s*INSUFFICIENT\s+FUNDS\s*-\s*',
        r'CASH\s*-\s*BNA\s*-\s*SELF\s*-\s*',
        r'Cash\s+Withdrawal\s*-\s*',
        r'ATM\s+WDL\s*-\s*ATM\s+CASH\s*\d*',
        r'ATM\s+CASH\s*\d*',
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

    # 3. Remove standard transaction prefixes with IDs:
    # e.g., "UPI/CR/414806262214/", "UPI/DR/123456789/", "RTGS Dr-CNRBR432998576731-", "NEFT Cr-468604178059-"
    text = re.sub(r'\b(UPI|RTGS|NEFT|IMPS)\s*/?\s*(Dr|Cr|DR|CR)?\s*[-/]?\s*[A-Z0-9]{8,24}\s*[-/]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(UPI|RTGS|NEFT|IMPS)\s*(Dr|Cr|DR|CR)?\s*[-/]\s*', ' ', text, flags=re.IGNORECASE)

    # 4. Remove standard Indian IFSC codes (4 letters + 0 + 6 alphanumeric characters, e.g. HDFC0004171, ICIC0SF0002)
    text = re.sub(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', ' ', text, flags=re.IGNORECASE)

    # 5. Remove long numeric IDs (reference numbers, phone numbers, UTRs >= 6 digits)
    text = re.sub(r'\b\d{6,}\b', ' ', text)

    # 6. Clean remaining delimiters, trailing slashes, dashes, and extra whitespace
    text = re.sub(r'[/\\|:_\-]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove leading/trailing non-word characters
    text = re.sub(r'^[^\w]+|[^\w]+$', '', text).strip()
    
    return text

def categorize_transaction(clean_text, raw_text):
    """Assigns high-level category based on clean meaningful text."""
    combined = (str(clean_text) + " " + str(raw_text)).upper()
    
    if "SALARY" in combined:
        return "Salary / Payroll"
    elif "RENT" in combined:
        return "Rent Payment"
    elif "SUBSCRIPTION" in combined or "SOFTWARE" in combined:
        return "Software & Subscriptions"
    elif any(k in combined for k in ["CASH", "ATM", "BNA", "WITHDRAWAL"]):
        return "Cash & ATM"
    elif any(k in combined for k in ["LTD", "CORP", "PVT", "ENTERPRISES", "INDUSTRIES", "TRADING", "SOLUTIONS", "INFOTECH", "HEALTHCARE", "PHARMA", "PROPERTIES"]):
        return "Vendor / Business"
    elif any(k in combined for k in ["CHQ", "CHEQUE", "CLEARING"]):
        return "Cheque Clearing"
    else:
        return "Personal / Transfer"

def filter_and_clean_data(df, target_col="Description"):
    """
    Filters and cleans the DataFrame:
    - Removes non-transaction header rows & empty rows
    - Extracts Cleaned_Description, Transaction_Channel, and Category
    """
    resolved_col = resolve_target_column(df, target_col)
    
    # 1. Filter out empty description rows & table headers
    is_invalid = (
        df[resolved_col].isna() |
        df[resolved_col].astype(str).str.strip().eq('') |
        df[resolved_col].astype(str).str.lower().str.strip().isin(['description', 'narration', 'particulars', 'nan']) |
        df[df.columns[0]].astype(str).str.lower().str.contains('txn date|date|account holders', regex=True, na=False)
    )
    
    clean_df = df[~is_invalid].copy().reset_index(drop=True)
    print(f"Filtered out {is_invalid.sum()} non-transaction / header rows. Remaining valid transactions: {len(clean_df)}")

    # 2. Extract meaningful information
    clean_df['Cleaned_Description'] = clean_df[resolved_col].apply(clean_description)
    clean_df['Transaction_Channel'] = clean_df[resolved_col].apply(extract_channel)
    clean_df['Category'] = clean_df.apply(
        lambda row: categorize_transaction(row['Cleaned_Description'], row[resolved_col]), axis=1
    )

    # 3. Filter out any remaining blank cleaned descriptions
    clean_df = clean_df[clean_df['Cleaned_Description'] != ''].reset_index(drop=True)

    return clean_df, resolved_col

def main():
    input_file = "extracted.xlsx"
    target_column = "Description"
    output_excel = "filtered_data.xlsx"
    output_csv = "filtered_data.csv"

    # 1. Load Data
    df, resolved_path = load_data(input_file)
    print(f"Initial shape: {df.shape}")

    # 2. Clean & Filter Data
    filtered_df, resolved_col = filter_and_clean_data(df, target_col=target_column)

    # 3. Display Sample Before vs After
    print("\n" + "=" * 110)
    print(f"{'RAW DESCRIPTION':<45} | {'CLEANED MEANINGFUL DATA':<32} | {'CHANNEL':<12} | {'CATEGORY'}")
    print("=" * 110)
    samples = filtered_df[[resolved_col, 'Cleaned_Description', 'Transaction_Channel', 'Category']].head(20)
    for _, row in samples.iterrows():
        raw = str(row[resolved_col]).replace('\n', ' ')[:43]
        clean = str(row['Cleaned_Description'])[:30]
        channel = str(row['Transaction_Channel'])[:10]
        cat = str(row['Category'])
        print(f"{raw:<45} | {clean:<32} | {channel:<12} | {cat}")
    print("=" * 110)

    # 4. Save Cleaned Dataset
    filtered_df.to_excel(output_excel, index=False)
    filtered_df.to_csv(output_csv, index=False)
    print(f"\nFiltered dataset saved to:")
    print(f"  - Excel: {output_excel}")
    print(f"  - CSV:   {output_csv}")

if __name__ == "__main__":
    main()
