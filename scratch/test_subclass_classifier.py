import os
import re
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ==============================================================================
# 1. TEXT CLEANING
# ==============================================================================
def clean_description(raw_text):
    if pd.isna(raw_text) or not str(raw_text).strip():
        return ""
    text = str(raw_text).strip()
    text = re.sub(r'\b(RAMES|MANIS|RAJES|HEALTHC|INVESTME|WELLNES)\s*\n\s*([A-Z]+)\b', r'\1\2', text, flags=re.IGNORECASE)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'/NONE\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'/URGENT/?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'SENT-TRANSFER\s+FROM\s+\d*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'UPI-TRANSFER\s+TO\s+\d*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'FROM\s+\d{6,}', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'TO\s+\d{6,}', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Za-z0-9]*\d+[A-Za-z0-9]*\b', ' ', text)
    text = re.sub(r'[-–—]+', '-', text)
    text = re.sub(r'[/\\|_]+', ' ', text)
    text = re.sub(r'(?:\s*-\s*)+', ' - ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip(' -:,./')

# ==============================================================================
# 2. SUB-TAXONOMY & EXEMPLARS
# ==============================================================================
SUB_TAXONOMY = {
    "Transaction Mode": {
        "UPI": ["UPI payment Unified Payments Interface instant mobile remittance", "UPI CR UPI DR UPI-TRANSFER mobile pay"],
        "NEFT": ["NEFT National Electronic Funds Transfer interbank payment electronic settlement", "NEFT Dr NEFT Cr"],
        "RTGS": ["RTGS Real Time Gross Settlement fund transfer high value payment", "RTGS Dr RTGS Cr"],
        "IMPS": ["IMPS Immediate Payment Service 24x7 instant remittance branch transfer", "IMPS Dr IMPS Cr IMPS BRN"],
        "Cash / BNA": ["CASH BNA Branch Cash Deposit Automated Machine transaction", "Cash Deposit BNA Deposit cash counter"],
        "ATM": ["ATM WDL ATM Cash automated teller machine cash withdrawal", "ATM Cash Withdrawal ATM WDL self cash"],
        "Cheque / Clearing": ["Chq Paid Cheque clearing MICR Inward Clearing cheque settlement", "By Clg Clearing Cheque clearing house outward clearing"],
        "NetBanking": ["Online Banking NetBanking digital internet transaction fee charges", "NetBanking portal web transaction"],
        "Bank Charges": ["Service Charges statement charges cheque book charges fee", "Maintenance charges annual fee SMS charges"]
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
        "Salary / Payroll": ["SALARY TRF Salary transfer monthly employee payroll remuneration staff pay wages bonus"],
        "Rent / Lease": ["RENT PAYMENT MONTHLY RENT house office rent lease payment commercial rent"],
        "Software / SaaS": ["SOFTWARE SUBSCRIPTION tech cloud software license SaaS billing IT tools"],
        "Utility / Bills": ["UTILITY PAYMENT electricity water gas internet telephone bill mobile recharge power bill"],
        "Bank Charges": [
            "Service Charges Online Banking Charges internet fee maintenance fee",
            "Service Charges Cheque Book Charges cheque book issue fee",
            "Service Charges Statement Charges bank statement request fee",
            "Service Charges Foreign Currency Markup FX international transaction fee"
        ],
        "Investment / Wealth": ["INVESTMENT SERVICES mutual funds securities wealth management trading stocks shares SIP"],
        "Healthcare / Medical": ["HEALTHCARE Medical Solutions remedies pharma wellness medicines health clinic doctor hospital"],
        "Daily Needs / Retail": ["DAILY NEEDS Retail Brands consumer essentials clothing trading products goods groceries supermarket shopping"]
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

def extract_bank_institution(text, emb, sub_embeddings):
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
        
    bank_sims = {label: float(np.max(np.dot(sub_embeddings["Bank / Institution"][label], emb))) for label in SUB_TAXONOMY["Bank / Institution"]}
    best_bank, bank_score = max(bank_sims.items(), key=lambda x: x[1])
    return best_bank if bank_score >= 0.65 else "Not Mentioned"

def extract_transaction_purpose(text, emb, sub_embeddings):
    txt_u = text.upper()
    if "SALARY" in txt_u or "PAYROLL" in txt_u:
        return "Salary / Payroll"
    if "RENT" in txt_u or "LEASE" in txt_u:
        return "Rent / Lease"
    if "SUBSCRIPTION" in txt_u or "SAAS" in txt_u or "SOFTWARE" in txt_u:
        return "Software / SaaS"
    if "UTILITY" in txt_u or "ELECTRICITY" in txt_u or "GAS" in txt_u or "WATER BILL" in txt_u:
        return "Utility / Bills"
    if "SERVICE CHARGES" in txt_u or "BANK CHARGES" in txt_u or "STATEMENT CHARGES" in txt_u or "MARKUP" in txt_u or "ONLINE BANKING CHARGES" in txt_u:
        return "Bank Charges"
    if "INVESTMENT" in txt_u or "MUTUAL FUND" in txt_u or "SECURITIES" in txt_u:
        return "Investment / Wealth"
    if "HEALTHCARE" in txt_u or "PHARMA" in txt_u or "REMEDIES" in txt_u or "MEDICAL" in txt_u:
        return "Healthcare / Medical"
    if "DAILY NEEDS" in txt_u or "RETAIL BRANDS" in txt_u or "CONSUMER ESSENTIALS" in txt_u or "GROCERIES" in txt_u:
        return "Daily Needs / Retail"
        
    purp_sims = {label: float(np.max(np.dot(sub_embeddings["Transaction Purpose"][label], emb))) for label in SUB_TAXONOMY["Transaction Purpose"]}
    best_purp, purp_score = max(purp_sims.items(), key=lambda x: x[1])
    return best_purp if purp_score >= 0.62 else "General Transfer"

def extract_counterparty(text):
    text_upper = text.upper()
    for ent in KNOWN_ENTITIES:
        if ent in text_upper:
            return ent.title()
    
    if re.search(r'\bSELF\b', text_upper):
        return "Self"
        
    # Heuristic name extraction
    segments = [s.strip() for s in re.split(r'[-–—:]', text) if s.strip()]
    for seg in reversed(segments):
        seg_clean = re.sub(r'\b(UPI|CR|DR|NEFT|RTGS|IMPS|CASH|BNA|ATM|WDL|CHQ|PAID|MICR|INWARD|CLEARING|SALARY|RENT|TRF|BY|TRANSFER)\b', '', seg, flags=re.IGNORECASE).strip()
        if len(seg_clean) > 3 and not re.search(r'(BANK|CHARGES|SERVICE|STATEMENT|SUBSCRIPTION|PAYMENT|ACCTS)', seg_clean, flags=re.IGNORECASE):
            return seg_clean.title()
    return "Not Specified"

def classify_row(text, model, sub_embeddings):
    clean_txt = clean_description(text)
    eval_txt = clean_txt if len(clean_txt) > 0 else text
    
    prefixed = f"search_document: {eval_txt}"
    emb = model.encode([prefixed], show_progress_bar=False)[0]
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    results = {}
    
    # 1. Transaction Mode
    txt_u = eval_txt.upper()
    if re.search(r'\bUPI\b', txt_u):
        results["Transaction Mode"] = "UPI"
    elif re.search(r'\bNEFT\b', txt_u):
        results["Transaction Mode"] = "NEFT"
    elif re.search(r'\bRTGS\b', txt_u):
        results["Transaction Mode"] = "RTGS"
    elif re.search(r'\bIMPS\b', txt_u):
        results["Transaction Mode"] = "IMPS"
    elif re.search(r'\b(CASH|BNA)\b', txt_u):
        results["Transaction Mode"] = "Cash / BNA"
    elif re.search(r'\bATM\b', txt_u):
        results["Transaction Mode"] = "ATM"
    elif re.search(r'\b(CHQ|CHEQUE|CLEARING|MICR|CLG)\b', txt_u):
        results["Transaction Mode"] = "Cheque / Clearing"
    elif re.search(r'\b(NETBANKING|ONLINE BANKING)\b', txt_u):
        results["Transaction Mode"] = "NetBanking"
    elif re.search(r'\b(SERVICE CHARGES|CHARGES|FEE)\b', txt_u):
        results["Transaction Mode"] = "Bank Charges"
    else:
        mode_sims = {label: float(np.max(np.dot(sub_embeddings["Transaction Mode"][label], emb))) for label in SUB_TAXONOMY["Transaction Mode"]}
        best_mode, mode_score = max(mode_sims.items(), key=lambda x: x[1])
        results["Transaction Mode"] = best_mode if mode_score >= 0.55 else "Other"

    # 2. Transaction Direction
    if re.search(r'\b(CR|CREDIT|DEPOSIT|INWARD)\b', txt_u) and not re.search(r'\b(DR|DEBIT|WDL|OUTWARD)\b', txt_u):
        results["Transaction Direction"] = "Credit"
    elif re.search(r'\b(DR|DEBIT|WDL|PAID|OUTWARD)\b', txt_u):
        results["Transaction Direction"] = "Debit"
    else:
        dir_sims = {label: float(np.max(np.dot(sub_embeddings["Transaction Direction"][label], emb))) for label in SUB_TAXONOMY["Transaction Direction"]}
        best_dir, dir_score = max(dir_sims.items(), key=lambda x: x[1])
        results["Transaction Direction"] = best_dir if dir_score >= 0.50 else "Other"

    # 3. Transaction Purpose
    results["Transaction Purpose"] = extract_transaction_purpose(eval_txt, emb, sub_embeddings)

    # 4. Bank / Institution
    results["Bank / Institution"] = extract_bank_institution(eval_txt, emb, sub_embeddings)

    # 5. Party / Counterparty
    results["Party / Counterparty"] = extract_counterparty(eval_txt)

    return results

if __name__ == "__main__":
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
    sub_embeddings = {}
    for cat, sub_dict in SUB_TAXONOMY.items():
        sub_embeddings[cat] = {}
        for sub_label, exemplars in sub_dict.items():
            pref = [f"search_document: {ex}" for ex in exemplars]
            embs = model.encode(pref, show_progress_bar=False)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            sub_embeddings[cat][sub_label] = embs / norms

    df = pd.read_excel("filtered_only description.xlsx")
    desc_col = df.columns[0]
    
    test_rows = df[desc_col].head(25).tolist()
    out_rows = []
    for t in test_rows:
        res = classify_row(t, model, sub_embeddings)
        out_rows.append({
            "Description": t,
            "Transaction Mode": res["Transaction Mode"],
            "Transaction Direction": res["Transaction Direction"],
            "Transaction Purpose": res["Transaction Purpose"],
            "Bank / Institution": res["Bank / Institution"],
            "Party / Counterparty": res["Party / Counterparty"]
        })
        
    res_df = pd.DataFrame(out_rows)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    print("\n--- REFINED CLASSIFICATION OUTPUT (NO SCORES) ---")
    print(res_df.to_string(index=False))
