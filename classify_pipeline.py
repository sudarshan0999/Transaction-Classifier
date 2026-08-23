import os
import re
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ==============================================================================
# 1. 5-CLASS SUB-TAXONOMY AND CANONICAL CATEGORY EXEMPLARS
# ==============================================================================
SUB_TAXONOMY = {
    "Transaction Mode": {
        "UPI": [
            "UPI payment Unified Payments Interface instant mobile remittance",
            "UPI CR UPI DR UPI-TRANSFER mobile barcode payment"
        ],
        "NEFT": [
            "NEFT National Electronic Funds Transfer interbank payment electronic settlement",
            "NEFT Dr NEFT Cr interbank remittance"
        ],
        "RTGS": [
            "RTGS Real Time Gross Settlement fund transfer high value payment",
            "RTGS Dr RTGS Cr high value settlement"
        ],
        "IMPS": [
            "IMPS Immediate Payment Service 24x7 instant remittance branch transfer",
            "IMPS Dr IMPS Cr IMPS BRN instant transfer"
        ],
        "ATM": [
            "ATM WDL ATM Cash automated teller machine cash withdrawal",
            "ATM Cash Withdrawal self cash payout ATM"
        ],
        "Cash / BNA": [
            "CASH BNA Branch Cash Deposit Automated Machine transaction",
            "Cash Deposit BNA Deposit cash counter branch teller"
        ],
        "Cheque / Clearing": [
            "Chq Paid Cheque clearing MICR Inward Clearing cheque settlement",
            "By Clg Clearing Cheque clearing house outward clearing"
        ],
        "NetBanking": [
            "Online Banking NetBanking digital internet transaction fee charges",
            "NetBanking portal web transaction fund transfer"
        ],
        "Bank Charges": [
            "Service Charges statement charges cheque book charges fee",
            "Maintenance charges annual fee SMS charges markup fee"
        ]
    },
    "Transaction Direction": {
        "Debit": [
            "Dr Debit outgoing payment money debited debit transfer",
            "UPI DR NEFT Dr RTGS Dr IMPS Dr Debit payment transfer out",
            "Cash Withdrawal ATM WDL funds withdrawn debit outflow payout",
            "Chq Paid Cheque Paid amount debited outward clearing"
        ],
        "Credit": [
            "Cr Credit incoming payment money credited credit received",
            "UPI CR NEFT Cr RTGS Cr IMPS Cr Credit received transfer in",
            "Cash Deposit BNA Deposit funds deposited credit inflow deposit",
            "MICR Inward Clearing By Clg credit incoming cheque clearing"
        ]
    },
    "Transaction Purpose": {
        "Salary / Payroll": [
            "SALARY TRF Salary transfer monthly employee payroll remuneration staff pay wages bonus",
            "Company salary disbursement monthly remuneration"
        ],
        "Rent / Lease": [
            "RENT PAYMENT MONTHLY RENT house office rent lease payment commercial rent",
            "Property rental accommodation lease payment"
        ],
        "Software / SaaS": [
            "SOFTWARE SUBSCRIPTION tech cloud software license SaaS billing IT tools web services",
            "Cloud hosting subscription digital license fee"
        ],
        "Utility / Bills": [
            "UTILITY PAYMENT electricity water gas internet telephone bill mobile recharge power bill",
            "Municipal water bill electricity electricity board payment"
        ],
        "Bank Charges": [
            "Service Charges Online Banking Charges internet fee maintenance fee",
            "Service Charges Cheque Book Charges cheque book issue fee",
            "Service Charges Statement Charges bank statement request fee",
            "Service Charges Foreign Currency Markup FX international transaction fee"
        ],
        "Investment / Wealth": [
            "INVESTMENT SERVICES mutual funds securities wealth management trading stocks shares SIP",
            "Portfolio investment capital market securities"
        ],
        "Healthcare / Medical": [
            "HEALTHCARE Medical Solutions remedies pharma wellness medicines health clinic doctor hospital",
            "Pharmacy pharmaceutical remedies diagnostic healthcare"
        ],
        "Daily Needs / Retail": [
            "DAILY NEEDS Retail Brands consumer essentials clothing trading products goods groceries supermarket",
            "Retail merchandise consumer products shopping purchase"
        ]
    },
    "Bank / Institution": {
        "HDFC Bank": ["HDFC BANK LTD commercial banking institution financial corporation"],
        "Citi Bank": ["CITI BANK N.A. CIT foreign bank financial institution"],
        "ICICI Bank": ["ICICI Bank Ltd commercial banking corporation"],
        "State Bank of India (SBI)": ["State Bank of India SBI nationalized bank public sector"],
        "Axis Bank": ["Axis Bank commercial banking corporation"],
        "Kotak Mahindra Bank": ["Kotak Mahindra Bank commercial banking private bank"],
        "Bank of Baroda": ["Bank of Baroda commercial banking public sector"],
        "Clearing House (MICR)": ["DEL ACCTS Clearing House MICR banking system institution inward outward clearing"],
        "Branch BNA": ["BNA Branch Network Automated Cash Machine bank branch teller"]
    }
}

# Known corporate & individual entities from statement datasets
KNOWN_ENTITIES = [
    "CONSUMER PRODUCTS INDIA", "DAILY NEEDS LIMITED", "WEBSTREAM LIMITED", "RETAIL BRANDS CORP",
    "COURIER NETWORKS INDIA", "COMMERCIAL PROPERTIES INDIA", "CONSUMER ESSENTIALS INDIA",
    "PROPERTY DEVELOPERS CORP", "FINANCIAL PRODUCTS LTD", "MEDICAL SOLUTIONS INDIA",
    "TRADING HOUSE LTD", "URBAN CONSTRUCTIONS", "NETFORGE TECHNOLOGIES", "PREMIUM PROPERTIES LTD",
    "FOUNDATION PROJECTS LTD", "DATABRIDGE INFOTECH", "FASHION TEXTILES LTD", "REMEDIES HEALTHCARE",
    "WELLNESS BRANDS INDIA", "HEALTHCARE PRODUCTS LTD", "CYBERCRAFT SYSTEMS", "ESTATE BUILDERS INDIA",
    "HEALTHPLUS PHARMA LTD", "GATEWAY PROJECTS LTD", "URBAN ESTATES INDIA", "MOTOR INDUSTRIES LTD",
    "SKYLINE BUILDERS", "MINDWAVE CONSULTING", "THERMAL SYSTEMS LTD", "MEDCARE LABORATORIES",
    "CLOUDTECH SERVICES", "VITALITY MEDICINES LTD", "CARGO SOLUTIONS LTD", "CLOTHING INDUSTRIES LTD",
    "CITYSCAPE DEVELOPERS", "MECHANICAL SOLUTIONS LTD", "MARKET DISTRIBUTORS CORP",
    "RESIDENTIAL PROJECTS LTD", "POWER SYSTEMS CORP", "LOAN FINANCE CORP", "COMMERCE SOLUTIONS LTD",
    "BUILDING SOLUTIONS LTD", "WELLNESS LABS INDIA",
    "RAMESH VORA", "NISHA JHAVERI", "KETAN MODI", "KAVITA AMIN", "SEJAL DAVE", "NILESH DOSHI",
    "NEHA PAREKH", "DHARA KOTHARI", "BHAVESH MODI", "RITU MEHTA", "RIDDHI BHATT", "ANJALI SHUKLA",
    "POOJA BHATT", "RAKESH VORA", "PRIYA VORA", "DINESH PAREKH", "KINJAL AMIN", "FORAM VYAS",
    "MANISH VYAS", "HIRAL PAREKH", "MANISH TRIVEDI", "SHRUTI MODI", "ASHISH THAKKAR", "MEHUL DAVE",
    "SURESH PAREKH", "SURESH DESAI", "MEHUL PATEL", "FORAM AMIN", "ANJALI DAVE", "RITU PATEL",
    "HITESH KOTHARI", "RAJESH VORA", "KAVITA GANDHI", "PRIYA MODI", "KAVITA DESAI", "ASHISH KOTHARI",
    "PRIYA MEHTA", "HIRAL JOSHI", "MEHUL PANDYA", "JAYESH VORA", "BHAVESH THAKKAR", "SEJAL PATEL",
    "KAVITA KOTHARI", "SHRUTI MEHTA", "ASHISH AMIN", "NISHA MEHTA", "PARESH PARIKH", "ANJALI PATEL",
    "FORAM JOSHI", "DHARA THAKKAR", "JAYESH MODI", "NEHA DESAI", "ASHISH DAVE", "NILESH PARIKH"
]


# ==============================================================================
# 2. DATA CLEANING UTILITY
# ==============================================================================
def clean_description(raw_text: str) -> str:
    """
    Cleans raw bank transaction descriptions by stripping UTRs, IFSC codes,
    account/phone numbers, and noise boilerplate while preserving keywords.
    """
    if pd.isna(raw_text) or not str(raw_text).strip():
        return ""
    
    text = str(raw_text).strip()
    
    # 1. Merge words split across line breaks
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

    # 3. Remove bank IFSC codes
    text = re.sub(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', ' ', text, flags=re.IGNORECASE)

    # 4. Remove all numeric tokens / reference IDs containing digits
    text = re.sub(r'\b[A-Za-z0-9]*\d+[A-Za-z0-9]*\b', ' ', text)

    # 5. Clean punctuation / redundant separators
    text = re.sub(r'[-–—]+', '-', text)
    text = re.sub(r'[/\\|_]+', ' ', text)
    text = re.sub(r'(?:\s*-\s*)+', ' - ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip(' -:,./')


# ==============================================================================
# 3. EXTRACTION HELPERS FOR 5 DOMAIN CLASSES
# ==============================================================================
def extract_transaction_mode(text: str, emb: np.ndarray, sub_embeddings: dict) -> str:
    txt_u = text.upper()
    if re.search(r'\bUPI\b', txt_u):
        return "UPI"
    elif re.search(r'\bNEFT\b', txt_u):
        return "NEFT"
    elif re.search(r'\bRTGS\b', txt_u):
        return "RTGS"
    elif re.search(r'\bIMPS\b', txt_u):
        return "IMPS"
    elif re.search(r'\bATM\b', txt_u):
        return "ATM"
    elif re.search(r'\b(CASH|BNA)\b', txt_u):
        return "Cash / BNA"
    elif re.search(r'\b(CHQ|CHEQUE|CLEARING|MICR|CLG)\b', txt_u):
        return "Cheque / Clearing"
    elif re.search(r'\b(NETBANKING|ONLINE BANKING)\b', txt_u):
        return "NetBanking"
    elif re.search(r'\b(SERVICE CHARGES|CHARGES|FEE)\b', txt_u):
        return "Bank Charges"
    
    # Semantic fallback
    mode_sims = {
        label: float(np.max(np.dot(sub_embeddings["Transaction Mode"][label], emb)))
        for label in SUB_TAXONOMY["Transaction Mode"]
    }
    best_mode, mode_score = max(mode_sims.items(), key=lambda x: x[1])
    return best_mode if mode_score >= 0.55 else "Other"


def extract_transaction_direction(text: str, emb: np.ndarray, sub_embeddings: dict, row: dict = None) -> str:
    """
    Classifies Transaction Direction (Debit vs Credit).
    1. Primary: Ground truth from ledger columns (Debit, Credit, Withdrawal, Deposit, Cr/Dr).
    2. Fallback: If amount columns are not available, uses regex heuristics and dense embeddings.
    """
    # ── Level 1: Ground Truth from Table Columns (if available) ───────────────
    if row is not None and isinstance(row, dict):
        def _has_val(keys):
            for k in keys:
                v = row.get(k)
                if v is not None and not pd.isna(v):
                    clean_v = str(v).replace("Rs.", "").replace(",", "").strip()
                    if clean_v not in ["", "0", "0.0", "0.00", "-", "nan", "None"]:
                        return True
            return False

        debit_present  = _has_val(["Debit", "Withdrawal", "Withdrawal Amt.", "Withdrawal Amount", "Dr", "Debit Amt", "Debit Amount"])
        credit_present = _has_val(["Credit", "Deposit", "Deposit Amt.", "Deposit Amount", "Cr", "Credit Amt", "Credit Amount"])

        if debit_present and not credit_present:
            return "Debit"
        if credit_present and not debit_present:
            return "Credit"

        # Check explicit Cr/Dr flags
        for flag_col in ["Cr/Dr", "CR/DR", "Direction", "Txn Type", "Type", "CR_DR"]:
            flag_val = str(row.get(flag_col, "")).strip().upper()
            if flag_val in ["CR", "CREDIT"]:
                return "Credit"
            if flag_val in ["DR", "DEBIT"]:
                return "Debit"

    # ── Level 2: Text Description Regex Heuristics ───────────────────────────
    txt_u = text.upper()
    if re.search(r'\b(CHQ PAID|CHEQUE PAID|ATM WDL|WITHDRAWAL|WDL|BILLPAY DR|UPI DR|NEFT DR|RTGS DR|IMPS DR)\b', txt_u):
        return "Debit"
    if re.search(r'\b(CR|CREDIT|DEPOSIT|INWARD|REFUND|SALARY TRF|UPI CR|NEFT CR|RTGS CR|IMPS CR)\b', txt_u) and not re.search(r'\b(DR|DEBIT|PAID|OUTWARD)\b', txt_u):
        return "Credit"
    if re.search(r'\b(DR|DEBIT|PAID|OUTWARD)\b', txt_u):
        return "Debit"

    # ── Level 3: Semantic Embedding Fallback (Cosine Similarity) ─────────────
    if emb is not None and sub_embeddings is not None and "Transaction Direction" in sub_embeddings:
        dir_sims = {
            label: float(np.max(np.dot(sub_embeddings["Transaction Direction"][label], emb)))
            for label in SUB_TAXONOMY["Transaction Direction"]
        }
        best_dir, dir_score = max(dir_sims.items(), key=lambda x: x[1])
        return best_dir if dir_score >= 0.50 else "Debit"

    return "Debit"


def extract_transaction_purpose(text: str, emb: np.ndarray, sub_embeddings: dict) -> str:
    txt_u = text.upper()
    if re.search(r'\b(SALARY|PAYROLL|WAGES|REMUNERATION)\b', txt_u):
        return "Salary / Payroll"
    if re.search(r'\b(RENT|LEASE)\b', txt_u):
        return "Rent / Lease"
    if re.search(r'\b(SUBSCRIPTION|SAAS|SOFTWARE)\b', txt_u):
        return "Software / SaaS"
    if re.search(r'\b(UTILITY|ELECTRICITY|GAS|WATER BILL|POWER BILL|TELEPHONE BILL)\b', txt_u):
        return "Utility / Bills"
    if re.search(r'\b(SERVICE CHARGES|BANK CHARGES|STATEMENT CHARGES|CHEQUE BOOK CHARGES|MARKUP|ONLINE BANKING CHARGES)\b', txt_u):
        return "Bank Charges"
    if re.search(r'\b(INVESTMENT|MUTUAL FUND|SECURITIES|STOCKS|SHARES|SIP)\b', txt_u):
        return "Investment / Wealth"
    if re.search(r'\b(HEALTHCARE|PHARMA|REMEDIES|MEDICAL|CLINIC|HOSPITAL)\b', txt_u):
        return "Healthcare / Medical"
    if re.search(r'\b(DAILY NEEDS|RETAIL BRANDS|CONSUMER ESSENTIALS|GROCERIES|SUPERMARKET)\b', txt_u):
        return "Daily Needs / Retail"
        
    # Semantic evaluation with strict threshold for purpose
    purp_sims = {
        label: float(np.max(np.dot(sub_embeddings["Transaction Purpose"][label], emb)))
        for label in SUB_TAXONOMY["Transaction Purpose"]
    }
    best_purp, purp_score = max(purp_sims.items(), key=lambda x: x[1])
    return best_purp if purp_score >= 0.62 else "General Transfer"


def extract_bank_institution(text: str, emb: np.ndarray, sub_embeddings: dict) -> str:
    txt_u = text.upper()
    if "HDFC" in txt_u:
        return "HDFC Bank"
    if "CITI" in txt_u or re.search(r'\bCIT\b', txt_u):
        return "Citi Bank"
    if "ICICI" in txt_u:
        return "ICICI Bank"
    if "SBI" in txt_u or "STATE BANK" in txt_u:
        return "State Bank of India (SBI)"
    if "AXIS" in txt_u:
        return "Axis Bank"
    if "KOTAK" in txt_u:
        return "Kotak Mahindra Bank"
    if "BARODA" in txt_u:
        return "Bank of Baroda"
    if "MICR" in txt_u or "DEL ACCTS" in txt_u or "CLEARING" in txt_u:
        return "Clearing House (MICR)"
    if "BNA" in txt_u:
        return "Branch BNA"
        
    bank_sims = {
        label: float(np.max(np.dot(sub_embeddings["Bank / Institution"][label], emb)))
        for label in SUB_TAXONOMY["Bank / Institution"]
    }
    best_bank, bank_score = max(bank_sims.items(), key=lambda x: x[1])
    return best_bank if bank_score >= 0.65 else "Not Mentioned"


def extract_counterparty(text: str) -> str:
    txt_u = text.upper()
    for ent in KNOWN_ENTITIES:
        if ent in txt_u:
            return ent.title()
    
    if re.search(r'\bSELF\b', txt_u):
        return "Self"
        
    # Heuristic segment parsing
    segments = [s.strip() for s in re.split(r'[-–—:]', text) if s.strip()]
    for seg in reversed(segments):
        seg_clean = re.sub(
            r'\b(UPI|CR|DR|NEFT|RTGS|IMPS|CASH|BNA|ATM|WDL|CHQ|PAID|MICR|INWARD|CLEARING|SALARY|RENT|TRF|BY|TRANSFER)\b',
            '', seg, flags=re.IGNORECASE
        ).strip()
        if len(seg_clean) > 3 and not re.search(r'(BANK|CHARGES|SERVICE|STATEMENT|SUBSCRIPTION|PAYMENT|ACCTS)', seg_clean, flags=re.IGNORECASE):
            return seg_clean.title()
            
    return "Not Specified"


# ==============================================================================
# 4. INITIALIZE OR LOAD MODEL & SUB-CATEGORY EMBEDDINGS
# ==============================================================================
def get_classification_engine(model_name="nomic-ai/nomic-embed-text-v1.5", cache_dir="embedding_class"):
    """
    Loads SentenceTransformer model and pre-computes sub-taxonomy embeddings.
    Caches the sub-category embeddings to disk for instantaneous loading.
    """
    model = SentenceTransformer(model_name, trust_remote_code=True)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sub_taxonomy_embeddings.npz")

    sub_embeddings = {}
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        for cat, sub_dict in SUB_TAXONOMY.items():
            sub_embeddings[cat] = {}
            for sub_label in sub_dict:
                key = f"{cat}__{sub_label}"
                if key in data:
                    sub_embeddings[cat][sub_label] = data[key]
    
    # Verify all keys exist
    missing = False
    for cat, sub_dict in SUB_TAXONOMY.items():
        if cat not in sub_embeddings:
            sub_embeddings[cat] = {}
        for sub_label in sub_dict:
            if sub_label not in sub_embeddings[cat]:
                missing = True
                break

    if missing:
        flat_save = {}
        for cat, sub_dict in SUB_TAXONOMY.items():
            sub_embeddings[cat] = {}
            for sub_label, exemplars in sub_dict.items():
                pref = [f"search_document: {ex}" for ex in exemplars]
                embs = model.encode(pref, show_progress_bar=False)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                norm_embs = (embs / norms).astype(np.float32)
                sub_embeddings[cat][sub_label] = norm_embs
                flat_save[f"{cat}__{sub_label}"] = norm_embs
        np.savez_compressed(cache_path, **flat_save)

    return model, sub_embeddings


# ==============================================================================
# 5. CORE CLASSIFICATION PIPELINE (NO SCORES WRITTEN)
# ==============================================================================
def classify_excel_dataframe(
    df: pd.DataFrame,
    model: SentenceTransformer,
    sub_embeddings: dict,
    desc_col: str = None
) -> pd.DataFrame:
    """
    Takes an input DataFrame, cleans transaction descriptions, dynamically generates embeddings,
    and appends 5 classified category columns with categorical values (WITHOUT score columns).
    """
    # 1. Resolve description column
    if desc_col is None or desc_col not in df.columns:
        candidates = ["description", "narration", "particulars", "transaction remarks", "details"]
        for col in df.columns:
            if str(col).strip().lower() in candidates:
                desc_col = col
                break
        if desc_col is None:
            desc_col = df.columns[0]

    raw_descriptions = df[desc_col].astype(str).tolist()
    cleaned_descriptions = [clean_description(d) for d in raw_descriptions]
    eval_texts = [c if len(c) > 0 else r for c, r in zip(cleaned_descriptions, raw_descriptions)]

    # 2. Generate embeddings for uploaded descriptions in batch
    prefixed_texts = [f"search_document: {t}" for t in eval_texts]
    embeddings = model.encode(prefixed_texts, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_embeddings = embeddings / norms

    # 3. Classify each row across the 5 domains
    modes = []
    directions = []
    purposes = []
    banks = []
    parties = []

    for i, eval_txt in enumerate(eval_texts):
        emb = norm_embeddings[i]
        row_dict = df.iloc[i].to_dict() if i < len(df) else None
        modes.append(extract_transaction_mode(eval_txt, emb, sub_embeddings))
        directions.append(extract_transaction_direction(eval_txt, emb, sub_embeddings, row=row_dict))
        purposes.append(extract_transaction_purpose(eval_txt, emb, sub_embeddings))
        banks.append(extract_bank_institution(eval_txt, emb, sub_embeddings))
        parties.append(extract_counterparty(eval_txt))

    # 4. Construct updated DataFrame with original columns + 5 classified columns
    output_df = df.copy()
    output_df["Transaction Mode"] = modes
    output_df["Transaction Direction"] = directions
    output_df["Transaction Purpose"] = purposes
    output_df["Bank / Institution"] = banks
    output_df["Party / Counterparty"] = parties

    return output_df


def classify_and_save_artifacts(
    df: pd.DataFrame,
    source_name: str = "bank_statement",
    output_base_dir: str = "output",
    model: SentenceTransformer = None,
    sub_embeddings: dict = None,
    desc_col: str = None
):
    """
    1. Cleans description narratives.
    2. Generates and normalizes dense embeddings.
    3. Categorizes 5 classes without score columns.
    4. Automatically stores embeddings .npy into output/embeddings/
    5. Automatically stores classified Excel into output/classified_statements/
    Returns (classified_df, embeddings_path, excel_path)
    """
    if model is None or sub_embeddings is None:
        model, sub_embeddings = get_classification_engine()

    # Create dedicated output subdirectories
    embeddings_dir = os.path.join(output_base_dir, "embeddings")
    statements_dir = os.path.join(output_base_dir, "classified_statements")
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(statements_dir, exist_ok=True)

    # Clean file identifier
    clean_stem = re.sub(r'[^A-Za-z0-9_\-]+', '_', os.path.splitext(source_name)[0]).strip('_')
    if not clean_stem:
        clean_stem = "statement"

    # 1. Resolve description column
    if desc_col is None or desc_col not in df.columns:
        candidates = ["description", "narration", "particulars", "transaction remarks", "details"]
        for col in df.columns:
            if str(col).strip().lower() in candidates:
                desc_col = col
                break
        if desc_col is None:
            desc_col = df.columns[0]

    raw_descriptions = df[desc_col].astype(str).tolist()
    cleaned_descriptions = [clean_description(d) for d in raw_descriptions]
    eval_texts = [c if len(c) > 0 else r for c, r in zip(cleaned_descriptions, raw_descriptions)]

    # 2. Compute embeddings
    prefixed_texts = [f"search_document: {t}" for t in eval_texts]
    embeddings = model.encode(prefixed_texts, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_embeddings = embeddings / norms

    # 3. Save description embeddings .npy in separate folder
    embeddings_path = os.path.join(embeddings_dir, f"{clean_stem}_description_embeddings.npy")
    np.save(embeddings_path, norm_embeddings)

    modes, directions, purposes, banks, parties = [], [], [], [], []
    for i, eval_txt in enumerate(eval_texts):
        emb = norm_embeddings[i]
        row_dict = df.iloc[i].to_dict() if i < len(df) else None
        modes.append(extract_transaction_mode(eval_txt, emb, sub_embeddings))
        directions.append(extract_transaction_direction(eval_txt, emb, sub_embeddings, row=row_dict))
        purposes.append(extract_transaction_purpose(eval_txt, emb, sub_embeddings))
        banks.append(extract_bank_institution(eval_txt, emb, sub_embeddings))
        parties.append(extract_counterparty(eval_txt))

    output_df = df.copy()
    output_df["Transaction Mode"] = modes
    output_df["Transaction Direction"] = directions
    output_df["Transaction Purpose"] = purposes
    output_df["Bank / Institution"] = banks
    output_df["Party / Counterparty"] = parties

    # 5. Save classified Excel in separate folder
    excel_path = os.path.join(statements_dir, f"{clean_stem}_classified_transactions.xlsx")
    output_df.to_excel(excel_path, index=False)

    return output_df, embeddings_path, excel_path


if __name__ == "__main__":
    test_excel = "filtered_only description.xlsx"
    if os.path.exists(test_excel):
        output_excel = "classified_descriptions_output.xlsx"
        res_df = process_uploaded_excel(test_excel, output_excel)
        print("\n--- SAMPLE PREVIEW (FIRST 15 ROWS) ---")
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 1000)
        preview_cols = ["Description", "Transaction Mode", "Transaction Direction", "Transaction Purpose", "Bank / Institution", "Party / Counterparty"]
        existing_preview_cols = [c for c in preview_cols if c in res_df.columns]
        print(res_df[existing_preview_cols].head(15).to_string(index=False))
