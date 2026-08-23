"""
new_training_data.py
====================
Generalized Training Pipeline for the Transaction Classification Engine.

Workflow:
  1. Scans all PDFs from the training_data/ folder.
  2. Extracts transactions using the app's schema-aware extraction engine.
  3. Deep-cleans all description text using enhanced cleaning rules.
  4. Derives real exemplar phrases from actual transaction data for all 5 domains.
  5. Regenerates all embeddings in embedding_class/ (overwrites old files).
  6. The updated embeddings are immediately used by the Streamlit app on next run.

Usage:
  python new_training_data.py
"""

import os
import re
import json
import sys
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
from collections import defaultdict

# ─── Import app's extraction engine ──────────────────────────────────────────
try:
    from app import extract_bank_statement
    from classify_pipeline import clean_description, SUB_TAXONOMY
    print("[OK] Successfully imported extraction engine and taxonomy from app.py")
except ImportError as e:
    print(f"[ERROR] Could not import from app.py: {e}")
    sys.exit(1)


# ─── Configuration ────────────────────────────────────────────────────────────
TRAINING_DATA_DIR   = "training_data"
EMBEDDING_CLASS_DIR = "embedding_class"
MODEL_NAME          = "nomic-ai/nomic-embed-text-v1.5"
CACHE_FILE          = os.path.join(EMBEDDING_CLASS_DIR, "sub_taxonomy_embeddings.npz")
METADATA_FILE       = os.path.join(EMBEDDING_CLASS_DIR, "class_metadata.json")

# Caps on real-data exemplars per label
MAX_REAL_EXEMPLARS_PER_LABEL = 6


# ==============================================================================
# 1. DEEP CLEANING
# ==============================================================================
_IFSC_RE      = re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', re.IGNORECASE)
_UTR_RE       = re.compile(r'\b\d{12,22}\b')
_REF_RE       = re.compile(r'\b(UTR|REF|TXN|TXNID|CRN|ARN|RRN|TRAN|CLRN)[\/\-\:]?\s*[A-Z0-9]{6,}\b', re.IGNORECASE)
_PHONE_RE     = re.compile(r'\b[6-9]\d{9}\b')
_ACCT_RE      = re.compile(r'\b\d{9,20}\b')
_MASKED_RE    = re.compile(r'\b[A-Z0-9]*X{4,}[A-Z0-9]*\b', re.IGNORECASE)
_VPA_RE       = re.compile(
    r'\S+@(?:ok|ybl|axl|hdfcbank|icici|paytm|okaxis|okbizaxis|okbiz|ibl|sbi|pnb|boi|'
    r'unionbank|federal|icicib|upi|apl|timecosmos|jupiter|slice|kbl|kvb|rbl|fbl|'
    r'naviaxis|yesbank|aubank|dbs|postbank|mahb|axis)\b', re.IGNORECASE)
_DATE_IN_DESC = re.compile(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b')
_NOISE_WORDS  = re.compile(
    r'\b(TO|FROM|BY|FOR|AT|ON|IN|OF|AND|OR|THE|A|AN|IS|WAS|ARE|WITH|TRF|TRANSFER|'
    r'TXN|PAYMENT|PAY|AMT|AMOUNT|RS|INR|BALANCE|BAL|NTNL|INTNL|CHARGES|CHARGE|FEE|FEES)\b',
    re.IGNORECASE
)


def deep_clean_description(raw: str) -> str:
    """
    Aggressive multi-stage cleaner for Indian bank narrations.
    Strips: UTRs, IFSC, VPA handles, phone/account numbers,
            masked card digits, reference codes, dates, noise filler words.
    Preserves: merchant names, bank keywords, payment mode identifiers.
    """
    if pd.isna(raw) or not str(raw).strip():
        return ""

    text = str(raw).strip().upper()
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)

    text = _VPA_RE.sub(' ', text)          # Remove VPA handles (@paytm, @okbiz ...)
    text = _IFSC_RE.sub(' ', text)         # Remove IFSC codes
    text = _REF_RE.sub(' ', text)          # Remove UTR / REF labels + codes
    text = _DATE_IN_DESC.sub(' ', text)    # Remove embedded dates
    text = _UTR_RE.sub(' ', text)          # Remove long numeric UTR strings
    text = _MASKED_RE.sub(' ', text)       # Remove masked card numbers XXXX
    text = _PHONE_RE.sub(' ', text)        # Remove 10-digit Indian mobile numbers
    text = _ACCT_RE.sub(' ', text)         # Remove remaining 9-20 digit account numbers

    # Remove transfer boilerplate
    text = re.sub(r'\b(SENT|RECEIVED|NONE|URGENT|NORMAL|NEFT|RTGS|IMPS|UPI|BILLPAY|IB|ACH|AUTO|DEBIT|CREDIT)\s+(TO|FROM)\b', ' ', text)

    text = re.sub(r'[-–—/\\|_:]+', ' ', text)   # Normalise separators
    text = _NOISE_WORDS.sub(' ', text)            # Remove filler words
    text = re.sub(r'[^A-Z0-9 ]', ' ', text)      # Keep only alphanumeric
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove tokens shorter than 2 chars
    tokens = [t for t in text.split() if len(t) >= 2]
    return ' '.join(tokens).lower().strip()


# ==============================================================================
# 2. DOMAIN SIGNAL EXTRACTORS
# ==============================================================================

def label_transaction_mode(desc_upper):
    if re.search(r'\bUPI\b', desc_upper):      return "UPI"
    if re.search(r'\bNEFT\b', desc_upper):     return "NEFT"
    if re.search(r'\bRTGS\b', desc_upper):     return "RTGS"
    if re.search(r'\bIMPS\b', desc_upper):     return "IMPS"
    if re.search(r'\bATM\b', desc_upper):      return "ATM"
    if re.search(r'\b(CASH|BNA)\b', desc_upper):                                          return "Cash / BNA"
    if re.search(r'\b(CHQ|CHEQUE|CLG|MICR|CLEARING|BY CLG)\b', desc_upper):              return "Cheque / Clearing"
    if re.search(r'\b(NETBANKING|IB BILL|BILLPAY|ONLINE BANK)\b', desc_upper):           return "NetBanking"
    if re.search(r'\b(SERVICE CHARGE|CHARGES|MARKUP|STATEMENT CHG|CHEQUE BOOK)\b', desc_upper): return "Bank Charges"
    return None


def label_transaction_direction(row):
    def has_val(keys):
        for k in keys:
            v = row.get(k)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            clean = str(v).replace("Rs.", "").replace(",", "").strip()
            if clean not in ["", "0", "0.0", "0.00", "-", "nan", "None"]:
                return True
        return False

    if has_val(["Debit", "Withdrawal", "Withdrawal Amt."]):  return "Debit"
    if has_val(["Credit", "Deposit",   "Deposit Amt."]):     return "Credit"
    flag = str(row.get("Cr/Dr", "")).strip().upper()
    if flag in ["CR", "CREDIT"]: return "Credit"
    if flag in ["DR", "DEBIT"]:  return "Debit"
    return None


def label_transaction_purpose(desc_upper):
    if re.search(r'\b(SALARY|PAYROLL|WAGES|REMUNERATION)\b', desc_upper):              return "Salary / Payroll"
    if re.search(r'\b(RENT|LEASE|RESIDENTIAL)\b', desc_upper):                         return "Rent / Lease"
    if re.search(r'\b(SOFTWARE|SAAS|SUBSCRIPTION|CLOUD|HOSTING)\b', desc_upper):       return "Software / SaaS"
    if re.search(r'\b(ELECTRICITY|WATER BILL|GAS BILL|TELEPHONE|MOBILE RECHARGE|BROADBAND|WIFI|UTILITY|MSEB|BESCOM|TNEB|BEST|BSES|TORRENT)\b', desc_upper): return "Utility / Bills"
    if re.search(r'\b(SERVICE CHARGE|MARKUP|CHEQUE BOOK|STATEMENT CHARGE|ANNUAL FEE|SMS CHARGE|PROCESSING FEE)\b', desc_upper): return "Bank Charges"
    if re.search(r'\b(MUTUAL FUND|SIP|NIFTY|SENSEX|INVESTMENT|SECURITIES|TRADING|DEMAT|STOCK|SHARES|ZERODHA|GROWW|UPSTOX)\b', desc_upper): return "Investment / Wealth"
    if re.search(r'\b(PHARMA|MEDICAL|HOSPITAL|CLINIC|DOCTOR|HEALTH|REMEDIES|APOLLO|MEDPLUS|NETMEDS|PHARMEASY)\b', desc_upper): return "Healthcare / Medical"
    if re.search(r'\b(GROCERY|SUPERMARKET|BIGBASKET|BLINKIT|ZEPTO|DMART|RELIANCE FRESH|GROFERS|SWIGGY INSTAMART|JIOMART)\b', desc_upper): return "Daily Needs / Retail"
    return None


def label_bank_institution(desc_upper):
    if "HDFC" in desc_upper:                                                              return "HDFC Bank"
    if "CITI" in desc_upper or re.search(r'\bCIT\b', desc_upper):                        return "Citi Bank"
    if "ICICI" in desc_upper:                                                             return "ICICI Bank"
    if re.search(r'\bSBI\b', desc_upper) or "STATE BANK" in desc_upper:                  return "State Bank of India (SBI)"
    if "AXIS" in desc_upper:                                                              return "Axis Bank"
    if "KOTAK" in desc_upper:                                                             return "Kotak Mahindra Bank"
    if "BARODA" in desc_upper or re.search(r'\bBOB\b', desc_upper):                      return "Bank of Baroda"
    if re.search(r'\bMICR\b', desc_upper) or "CLEARING" in desc_upper or "CLG" in desc_upper: return "Clearing House (MICR)"
    if re.search(r'\bBNA\b', desc_upper):                                                 return "Branch BNA"
    return None


# ==============================================================================
# 3. EXTRACT REAL EXEMPLARS FROM DATA
# ==============================================================================

# Minimum quality standards for a real exemplar to enter training buckets
_MIN_EXEMPLAR_TOKENS = 3     # At least 3 meaningful words
_MIN_EXEMPLAR_CHARS  = 12    # At least 12 characters
# Patterns that mark an exemplar as noise (pure reference codes, fragments, etc.)
_NOISE_EXEMPLAR_RE = re.compile(
    r'^(\d+\s+upi|collect request|esb|hdfc0|upi$|\d+$)',
    re.IGNORECASE
)


def _is_quality_exemplar(text: str) -> bool:
    """Returns True only if the phrase is a meaningful semantic exemplar, not noise."""
    if len(text) < _MIN_EXEMPLAR_CHARS:
        return False
    tokens = [t for t in text.split() if len(t) >= 2 and not t.isdigit()]
    if len(tokens) < _MIN_EXEMPLAR_TOKENS:
        return False
    if _NOISE_EXEMPLAR_RE.match(text.strip()):
        return False
    # Reject if more than 50% of tokens are numeric
    all_tokens = text.split()
    numeric_ratio = sum(1 for t in all_tokens if re.match(r'^\d+$', t)) / max(len(all_tokens), 1)
    if numeric_ratio > 0.5:
        return False
    return True


def label_transaction_purpose(desc_upper):
    """Returns a Transaction Purpose label from raw uppercase description, or None."""
    # ATM first — unambiguous
    if re.search(r'\b(ATM WDL|ATM WITHDRAWAL|ATM CASH)\b', desc_upper):         return "ATM Withdrawal"
    # Bank charges
    if re.search(r'\b(SERVICE CHARGES|BANK CHARGES|STATEMENT CHARGES|CHEQUE BOOK CHARGES|MARKUP|ANNUAL FEE|SMS CHARGES)\b', desc_upper): return "Bank Charges"
    # Salary
    if re.search(r'\b(SALARY|PAYROLL|WAGES|REMUNERATION|IMPS BRN SALARY)\b', desc_upper):   return "Salary / Payroll"
    # Investment
    if re.search(r'\b(MUTUAL FUND|SIP|DIVIDEND|SECURITIES|TRADING|DEMAT|ZERODHA|GROWW|EQUITY SHARES|STOCKS|SHARES)\b', desc_upper): return "Investment / Wealth"
    # Healthcare — specific brands only
    if re.search(r'\b(APOLLO PHARMACY|MEDPLUS|NETMEDS|PHARMEASY|HEALTHKART|DIAGNOSTIC|HOSPITAL|NURSING HOME|PATHOLOGY)\b', desc_upper): return "Healthcare / Medical"
    # Food & Dining — specific brands/keywords
    if re.search(r'\b(ZOMATO|SWIGGY|AMUL|ZEROX|PARLOUR|BAKERY|JUICE|RESTAURANT|CAFE|CANTEEN|DHABA|PIZZA|BURGER|DOMINOES|MCDONALDS|STARBUCKS)\b', desc_upper): return "Food & Dining"
    # Utility bills
    if re.search(r'\b(ELECTRICITY|MSEB|BESCOM|TNEB|TORRENT POWER|TPDDL|BSES|WATER BILL|GAS BILL|BROADBAND|JIOFIBER|BSNL|AIRTEL FIBER|DTH RECHARGE|MOBILE RECHARGE)\b', desc_upper): return "Utility / Bills"
    # Rent
    if re.search(r'\b(RENT PAYMENT|MONTHLY RENT|HOUSE RENT|OFFICE RENT|LEASE PAYMENT|LANDLORD|RENTAL DEPOSIT)\b', desc_upper): return "Rent / Lease"
    # Software / SaaS
    if re.search(r'\b(SUBSCRIPTION|SAAS|SOFTWARE LICENSE|CLOUD HOSTING|AWS|AZURE|GOOGLE CLOUD|ADOBE|GITHUB|NETFLIX|HOTSTAR|PRIME VIDEO|SPOTIFY)\b', desc_upper): return "Software / SaaS"
    # Daily Needs
    if re.search(r'\b(BIGBASKET|BLINKIT|ZEPTO|DUNZO|DMART|RELIANCE FRESH|JIOMART|GROFERS)\b', desc_upper): return "Daily Needs / Retail"
    # Travel
    if re.search(r'\b(UBER|OLA CAB|RAPIDO|IRCTC|MAKEMYTRIP|YATRA|OYO|GOIBIBO|PETROL|FUEL STATION)\b', desc_upper): return "Travel & Transport"
    # P2P: only if clearly a person-name UPI transfer
    upi_name_m = re.match(r'UPI-([A-Z][A-Z ]{3,30})-', desc_upper)
    if upi_name_m:
        candidate = upi_name_m.group(1).strip()
        words = candidate.split()
        is_person = (
            len(words) >= 2
            and all(re.match(r'^[A-Z]+$', w) for w in words)
            and not re.search(r'\b(GOOGLE|AMAZON|FLIPKART|PAYTM|HDFC|ICICI|AXIS|SBI|BANK|STORE|SHOP|MART|INDIA|PVT|LTD|CORP|ZEROX|AMUL|BLINKIT|ZOMATO|SWIGGY)\b', candidate)
        )
        if is_person:
            return "Personal / P2P Transfer"
    return None


def extract_real_exemplars(all_rows):
    buckets = {
        "Transaction Mode":      defaultdict(set),
        "Transaction Direction": defaultdict(set),
        "Transaction Purpose":   defaultdict(set),
        "Bank / Institution":    defaultdict(set),
    }

    for row in all_rows:
        raw_desc = str(row.get("Description", "") or row.get("Narration", "") or "")
        if not raw_desc.strip():
            continue

        cleaned = deep_clean_description(raw_desc)
        # Quality gate: reject noisy/short cleaned text before it enters training
        if not _is_quality_exemplar(cleaned):
            continue

        desc_upper = raw_desc.upper()

        mode = label_transaction_mode(desc_upper)
        if mode:
            buckets["Transaction Mode"][mode].add(cleaned)

        direction = label_transaction_direction(row)
        if direction:
            buckets["Transaction Direction"][direction].add(cleaned)

        purpose = label_transaction_purpose(desc_upper)
        if purpose:
            buckets["Transaction Purpose"][purpose].add(cleaned)

        bank = label_bank_institution(desc_upper)
        if bank:
            buckets["Bank / Institution"][bank].add(cleaned)

    real_exemplars = {}
    for domain, label_dict in buckets.items():
        real_exemplars[domain] = {}
        for label, phrases_set in label_dict.items():
            # Only keep quality-filtered exemplars, sorted by length desc (richer ones first)
            quality_phrases = sorted(
                [p for p in phrases_set if _is_quality_exemplar(p)],
                key=len, reverse=True
            )[:MAX_REAL_EXEMPLARS_PER_LABEL]
            if quality_phrases:
                real_exemplars[domain][label] = quality_phrases

    return real_exemplars


def merge_exemplars(base_taxonomy, real_exemplars):
    merged = {}
    for domain, label_dict in base_taxonomy.items():
        merged[domain] = {}
        for label, base_phrases in label_dict.items():
            real = real_exemplars.get(domain, {}).get(label, [])
            combined = list(base_phrases) + [p for p in real if p not in base_phrases]
            merged[domain][label] = combined

    # Add new labels discovered from real data
    for domain, label_dict in real_exemplars.items():
        if domain not in merged:
            merged[domain] = {}
        for label, phrases in label_dict.items():
            if label not in merged[domain]:
                print(f"    [NEW LABEL] {domain} -> '{label}' ({len(phrases)} exemplars)")
                merged[domain][label] = phrases

    return merged


# ==============================================================================
# 4. EMBEDDING GENERATION
# ==============================================================================

def encode_and_normalize(model, texts):
    prefixed = [f"search_document: {t}" for t in texts]
    embs = model.encode(prefixed, show_progress_bar=False, batch_size=32)
    embs = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def build_and_save_embeddings(model, merged_taxonomy, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    class_prototypes  = []
    class_names       = []
    bundle_dict       = {}
    runtime_cache     = {}
    metadata_classes  = {}

    domain_id_map = {
        "Transaction Mode":      1,
        "Transaction Direction": 2,
        "Transaction Purpose":   3,
        "Bank / Institution":    4,
    }

    print(f"\n{'='*65}")
    print("GENERATING EMBEDDINGS FOR ALL DOMAINS")
    print(f"{'='*65}")

    for domain, label_dict in merged_taxonomy.items():
        dom_id     = domain_id_map.get(domain, 9)
        folder_key = f"{dom_id}_{domain.lower().replace(' / ', '_').replace(' ', '_').replace('/', '_')}"

        print(f"\n[Domain {dom_id}] {domain}")
        all_emb_rows = []
        domain_meta  = {"id": dom_id, "folder_key": folder_key, "labels": {}}

        for label, phrases in label_dict.items():
            if not phrases:
                print(f"    [SKIP] Label '{label}' has no exemplars.")
                continue

            print(f"    -> '{label}': {len(phrases)} exemplars")
            label_embs = encode_and_normalize(model, phrases)   # (n, 768)

            label_centroid = label_embs.mean(axis=0)
            nc = np.linalg.norm(label_centroid)
            if nc > 0:
                label_centroid /= nc

            all_emb_rows.append(label_centroid)

            # Per-label embedding matrix for sub_taxonomy_embeddings.npz
            key = f"{domain}__{label}"
            runtime_cache[key] = label_embs
            domain_meta["labels"][label] = phrases

        if not all_emb_rows:
            print(f"    [SKIP] No data for domain '{domain}'.")
            continue

        domain_matrix   = np.stack(all_emb_rows, axis=0)   # (k, 768)
        domain_centroid = domain_matrix.mean(axis=0)
        nc = np.linalg.norm(domain_centroid)
        if nc > 0:
            domain_centroid /= nc

        npy_path      = os.path.join(output_dir, f"{folder_key}.npy")
        centroid_path = os.path.join(output_dir, f"{folder_key}_centroid.npy")
        np.save(npy_path,      domain_matrix)
        np.save(centroid_path, domain_centroid)
        print(f"    Saved: {os.path.basename(npy_path)}  {domain_matrix.shape}")
        print(f"    Saved: {os.path.basename(centroid_path)}  {domain_centroid.shape}")

        bundle_dict[f"{folder_key}_exemplars"] = domain_matrix
        bundle_dict[f"{folder_key}_centroid"]  = domain_centroid
        class_prototypes.append(domain_centroid)
        class_names.append(domain)
        metadata_classes[domain] = domain_meta

    if class_prototypes:
        proto_arr = np.stack(class_prototypes, axis=0)
        np.save(os.path.join(output_dir, "class_prototypes.npy"), proto_arr)
        print(f"\nSaved: class_prototypes.npy  {proto_arr.shape}")

    np.savez_compressed(os.path.join(output_dir, "class_embeddings_bundle.npz"), **bundle_dict)
    print(f"Saved: class_embeddings_bundle.npz  ({len(bundle_dict)} arrays)")

    cache_path = os.path.join(output_dir, "sub_taxonomy_embeddings.npz")
    np.savez_compressed(cache_path, **runtime_cache)
    print(f"Saved: sub_taxonomy_embeddings.npz  ({len(runtime_cache)} label arrays)  <- USED BY APP")

    metadata = {
        "model_name":  MODEL_NAME,
        "trained_on":  "real_data + base_taxonomy_merged",
        "classes":     metadata_classes,
        "class_names": class_names,
    }
    with open(os.path.join(output_dir, "class_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"Saved: class_metadata.json")


# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    print("=" * 65)
    print("  TRANSACTION CLASSIFIER — TRAINING PIPELINE")
    print("  new_training_data.py")
    print("=" * 65)

    # Step 0: Validate training_data/
    if not os.path.isdir(TRAINING_DATA_DIR):
        os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
        print(f"\n[CREATED] '{TRAINING_DATA_DIR}/' folder.")
        print(f"  -> Place bank statement PDFs into '{TRAINING_DATA_DIR}/' and re-run.")
        sys.exit(0)

    pdf_files = sorted(Path(TRAINING_DATA_DIR).glob("*.pdf"))
    if not pdf_files:
        print(f"\n[WARNING] No PDF files found in '{TRAINING_DATA_DIR}/'.")
        print("  -> Copy bank statement PDFs into training_data/ and re-run.")
        sys.exit(0)

    print(f"\n[STEP 1] Found {len(pdf_files)} PDF(s) in '{TRAINING_DATA_DIR}/':")
    for pf in pdf_files:
        print(f"         • {pf.name}")

    # Step 1: Extract all transactions
    print(f"\n[STEP 2] Extracting transactions from all PDFs...")
    all_rows = []
    extraction_summary = []

    for pdf_path in pdf_files:
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            df, meta, _ = extract_bank_statement(pdf_bytes)

            if df is None or df.empty:
                print(f"  [SKIP] {pdf_path.name}: no transactions extracted.")
                continue

            rows = df.to_dict("records")
            all_rows.extend(rows)
            bank   = meta.get("Bank Name", "Unknown")
            holder = meta.get("Account Holder", "")
            extraction_summary.append({
                "file":  pdf_path.name,
                "bank":  bank,
                "holder": holder,
                "rows":  len(rows),
            })
            print(f"  [OK] {pdf_path.name}: {len(rows)} rows  [{bank} | {holder}]")

        except Exception as e:
            print(f"  [ERROR] {pdf_path.name}: {e}")

    if not all_rows:
        print("\n[ABORT] No transactions could be extracted from training PDFs.")
        sys.exit(1)

    print(f"\n  Total rows extracted: {len(all_rows)}")

    # Step 2: Sample of deep cleaning
    print(f"\n[STEP 3] Deep cleaning descriptions — sample:")
    for row in all_rows[:4]:
        raw   = str(row.get("Description", "") or "")
        clean = deep_clean_description(raw)
        print(f"  RAW  : {raw[:70]}")
        print(f"  CLEAN: {clean[:70]}")
        print()

    # Step 3: Extract real exemplars
    print(f"[STEP 4] Extracting real-data exemplars per domain label...")
    real_exemplars = extract_real_exemplars(all_rows)

    for domain, label_dict in real_exemplars.items():
        print(f"\n  {domain}:")
        for label, phrases in label_dict.items():
            print(f"    [{label}] {len(phrases)} exemplars")
            for p in phrases[:2]:
                print(f"       • {p}")

    # Step 4: Merge taxonomy + real exemplars
    print(f"\n[STEP 5] Merging base taxonomy with real-data exemplars...")
    merged = merge_exemplars(SUB_TAXONOMY, real_exemplars)
    for domain, label_dict in merged.items():
        total = sum(len(v) for v in label_dict.values())
        print(f"  {domain}: {len(label_dict)} labels, {total} total exemplars")

    # Step 5: Backup old embeddings
    backup_dir = EMBEDDING_CLASS_DIR + "_backup"
    if os.path.isdir(EMBEDDING_CLASS_DIR):
        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(EMBEDDING_CLASS_DIR, backup_dir)
        print(f"\n[BACKUP] Previous embeddings backed up to '{backup_dir}/'")

    try:
        # Step 6: Load model
        print(f"\n[STEP 6] Loading embedding model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        print(f"  Model ready. Embedding dimensions: 768")

        # Step 7: Build & save new embeddings
        print(f"\n[STEP 7] Generating and saving new embeddings...")
        build_and_save_embeddings(model, merged, EMBEDDING_CLASS_DIR)
    except Exception as e:
        print(f"\n[ERROR] Embedding generation failed: {e}")
        if os.path.isdir(backup_dir):
            print(f"[RESTORING] Rolling back to previous embeddings from '{backup_dir}/'...")
            if os.path.isdir(EMBEDDING_CLASS_DIR):
                shutil.rmtree(EMBEDDING_CLASS_DIR)
            shutil.copytree(backup_dir, EMBEDDING_CLASS_DIR)
            print(f"[RESTORED] Successfully rolled back to working backup in '{EMBEDDING_CLASS_DIR}/'.")
        raise e

    # Final report
    print(f"\n{'='*65}")
    print("  TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"  PDFs processed  : {len(pdf_files)}")
    print(f"  Total rows      : {len(all_rows)}")
    print(f"  Embeddings dir  : {os.path.abspath(EMBEDDING_CLASS_DIR)}/")
    print(f"\n  Extraction Summary:")
    for s in extraction_summary:
        print(f"    • {s['file']:<35} {s['rows']:>4} rows   [{s['bank']}]")
    print(f"\n  The Streamlit app will use the new embeddings automatically.\n")


if __name__ == "__main__":
    main()
