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
        "Personal / P2P Transfer": [
            "UPI transfer to individual person friend family relative personal send money peer to peer",
            "Personal payment to individual private transfer friend colleague known person",
            "Money sent to person private individual self transfer family member"
        ],
        "Salary / Payroll": [
            "SALARY TRF Salary transfer monthly employee payroll remuneration staff pay wages bonus",
            "Company salary disbursement monthly remuneration IMPS BRN SALARY",
            "Employee salary credit payroll disbursement wages monthly salary"
        ],
        "Rent / Lease": [
            "RENT PAYMENT MONTHLY RENT house office rent lease payment commercial rent",
            "Property rental accommodation lease payment landlord house rent"
        ],
        "Software / SaaS": [
            "SOFTWARE SUBSCRIPTION tech cloud software license SaaS billing IT tools web services",
            "Cloud hosting subscription digital license Google Apple Microsoft Adobe fee"
        ],
        "Utility / Bills": [
            "UTILITY PAYMENT electricity water gas internet broadband telephone bill mobile recharge power",
            "Electricity board MSEB BESCOM TNEB TORRENT TPDDL power bill payment BSNL Jio Airtel recharge"
        ],
        "Bank Charges": [
            "Service Charges Online Banking Charges internet fee maintenance fee annual charges SMS",
            "Service Charges Cheque Book Statement Charges Foreign Currency Markup fee penalty"
        ],
        "Investment / Wealth": [
            "INVESTMENT SERVICES mutual funds securities wealth management trading stocks shares SIP Zerodha Groww",
            "Portfolio investment capital market securities dividend equity demat brokerage"
        ],
        "Healthcare / Medical": [
            "HEALTHCARE Medical pharma wellness medicines clinic doctor hospital Apollo MedPlus pharmacy",
            "Pharmacy diagnostic lab health centre remedies medical bill doctor consultation"
        ],
        "Food & Dining": [
            "Zomato Swiggy food delivery restaurant dining cafe coffee bakery canteen mess snacks fast food",
            "Restaurant hotel food beverage juice centre Amul parlour tea stall eatery meal takeaway"
        ],
        "Daily Needs / Retail": [
            "DAILY NEEDS groceries supermarket consumer essentials Blinkit Zepto Dunzo BigBasket DMart",
            "Retail shopping clothing product purchase department store mall consumer goods household"
        ],
        "ATM Withdrawal": [
            "ATM cash withdrawal automated teller machine self cash payout ATM WDL",
            "ATM withdrawal cash self branch ATM machine money withdrawn"
        ],
        "Travel & Transport": [
            "Uber Ola cab taxi auto IRCTC railway flight airline hotel MakeMyTrip petrol fuel booking",
            "Travel transport logistics cab ride airline ticket hotel stay bus KSRTC GSRTC"
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


def extract_transaction_purpose(text: str, emb: np.ndarray, sub_embeddings: dict, raw_text: str = None) -> str:
    """
    Classifies Transaction Purpose across 12 categories using a 3-tier strategy:
      Tier 1: Unambiguous keyword/merchant matching on BOTH raw and cleaned text.
      Tier 2: Person-name pattern detection on raw text for P2P (requires UPI- prefix).
      Tier 3: Semantic embedding similarity with a strict 0.75 confidence threshold.

    Args:
      text:     The CLEANED description (used for embedding lookup).
      emb:      The embedding vector of the cleaned text.
      raw_text: The RAW description (used for Tier 1/2 regex, if available).
    """
    # Use raw text for keyword/pattern matching when available (preserves UPI- prefix, merchant names)
    match_text = raw_text if raw_text else text
    txt_u = match_text.upper()

    # ── Tier 1A: ATM Withdrawal ─────────────────────────────────────────────
    if re.search(r'\b(ATM WDL|ATM WITHDRAWAL|ATM CASH|ATM-WDL)\b', txt_u):
        return "ATM Withdrawal"

    # ── Tier 1B: Bank Charges (unambiguous fee keywords) ────────────────────
    if re.search(r'\b(SERVICE CHARGES|BANK CHARGES|STATEMENT CHARGES|CHEQUE BOOK CHARGES|ANNUAL FEE|SMS CHARGES|MARKUP|ONLINE BANKING CHARGES|PROCESSING FEE|ACCOUNT MAINTENANCE)\b', txt_u):
        return "Bank Charges"

    # ── Tier 1C: Salary (unambiguous payroll keywords) ──────────────────────
    if re.search(r'\b(SALARY|PAYROLL|WAGES|REMUNERATION|SALARY TRF|IMPS BRN SALARY|NEFT SALARY)\b', txt_u):
        return "Salary / Payroll"

    # ── Tier 1D: Investment (clear financial instrument keywords) ───────────
    if re.search(r'\b(MUTUAL FUND|SECURITIES|SIP|ZERODHA|GROWW|UPSTOX|TRADING|DEMAT|DIVIDEND|EQUITY SHARES|NIFTY|SENSEX|STOCKS|SHARES|NSE|BSE)\b', txt_u):
        return "Investment / Wealth"

    # ── Tier 1E: Healthcare (specific pharmacy/clinic brands & terms) ────────
    if re.search(r'\b(APOLLO PHARMACY|MEDPLUS|NETMEDS|PHARMEASY|TATA MED|HEALTHKART|WELLNESS FOREVER|DIAGNOSTIC|PATHOLOGY|SCAN CENTRE|HOSPITAL|NURSING HOME)\b', txt_u):
        return "Healthcare / Medical"

    # ── Tier 1F: Food & Dining (specific food brands) ───────────────────────
    if re.search(r'\b(ZOMATO|SWIGGY|BLINKIT FOOD|AMUL|ZEROX|PARLOUR|BAKERY|JUICE|RESTAURANT|CAFE|CANTEEN|DHABA|HOTEL FOOD|SNACKS|PIZZA|BURGER|DOMINOES|MCDONALDS|STARBUCKS|CHAAYOS)\b', txt_u):
        return "Food & Dining"

    # ── Tier 1G: Utility/Bills (specific board/provider names) ──────────────
    if re.search(r'\b(ELECTRICITY|MSEB|BESCOM|TNEB|TORRENT POWER|TPDDL|BSES|BEST ELECTRIC|WATER BILL|GAS BILL|PIPED GAS|BROADBAND|WIFI BILL|TELEPHONE BILL|JIOFIBER|BSNL|AIRTEL FIBER|MOBILE RECHARGE|DTH RECHARGE)\b', txt_u):
        return "Utility / Bills"

    # ── Tier 1H: Rent / Lease (specific property terms) ─────────────────────
    if re.search(r'\b(RENT PAYMENT|MONTHLY RENT|HOUSE RENT|OFFICE RENT|PROPERTY RENT|LEASE PAYMENT|LANDLORD|RENTAL DEPOSIT|ACCOMMODATION)\b', txt_u):
        return "Rent / Lease"

    # ── Tier 1I: Software / SaaS (specific tech brands) ─────────────────────
    if re.search(r'\b(SUBSCRIPTION|SAAS|SOFTWARE LICENSE|CLOUD HOSTING|AWS|GOOGLE CLOUD|AZURE|ADOBE|GITHUB|CANVA|NETFLIX|HOTSTAR|PRIME VIDEO|SPOTIFY|DROPBOX|NOTION|SLACK|ZOOM)\b', txt_u):
        return "Software / SaaS"

    # ── Tier 1J: Daily Needs / Retail (grocery & hypermarket brands) ─────────
    if re.search(r'\b(BIGBASKET|BLINKIT|ZEPTO|DUNZO|DMART|RELIANCE FRESH|JIOMART|GROFERS|SWIGGY INSTAMART|MORE RETAIL|SPAR|NATURA|HYPERCITY|WALMART|COSTCO)\b', txt_u):
        return "Daily Needs / Retail"

    # ── Tier 1K: Travel & Transport ──────────────────────────────────────────
    if re.search(r'\b(UBER|OLA CAB|RAPIDO|IRCTC|RAILWAY|AIRLINE|SPICEJET|INDIGO|AIR INDIA|MAKEMYTRIP|YATRA|OYO|GOIBIBO|PETROL|FUEL STATION|SHELL|BHARAT PETROLEUM|HP PETROL)\b', txt_u):
        return "Travel & Transport"

    # ── Tier 2: Person-name pattern detection → P2P Transfer ────────────────
    # Works on raw text which preserves the full UPI-NAME-vpa@bank structure.
    # UPI narration format: UPI-FIRSTNAME LASTNAME[-vpa@bank-ifsc]
    upi_name_m = re.match(r'UPI-([A-Z][A-Z ]{3,35}?)(?:-[A-Z0-9@.]+|-[A-Z0-9]{2,}@|@|$)', txt_u)
    if upi_name_m:
        candidate = upi_name_m.group(1).strip()
        words = candidate.split()
        is_person = (
            len(words) >= 2
            and all(re.match(r'^[A-Z]+$', w) for w in words)
            and not re.search(
                r'\b(GOOGLE|AMAZON|FLIPKART|PAYTM|PHONE|HDFC|ICICI|AXIS|SBI|BANK|STORE|SHOP|MART|INDIA|PVT|LTD|CORP|ZEROX|AMUL|BLINKIT|ZOMATO|SWIGGY|DIGITAL|INDIA|SERVICES|PAYMENT|GATEWAY|NETWORK|SOLUTIONS)\b',
                candidate
            )
        )
        if is_person:
            return "Personal / P2P Transfer"

    # Non-UPI RTGS/NEFT to individuals: prefix patterns like MR FIRSTNAME LASTNAME
    if re.search(r'\b(MR|MS|MRS|DR|SHRI|SHREE|KU|MISS)\s+[A-Z]{2,}\s+[A-Z]{2,}\b', txt_u):
        return "Personal / P2P Transfer"

    # ── Tier 3: Semantic embedding fallback with strict threshold ────────────
    if emb is not None and sub_embeddings is not None and "Transaction Purpose" in sub_embeddings:
        purp_sims = {
            label: float(np.max(np.dot(sub_embeddings["Transaction Purpose"][label], emb)))
            for label in SUB_TAXONOMY["Transaction Purpose"]
        }
        best_purp, purp_score = max(purp_sims.items(), key=lambda x: x[1])
        # High threshold 0.75 to avoid false positives on ambiguous text
        return best_purp if purp_score >= 0.75 else "General Transfer"

    return "General Transfer"

    # ── Tier 1A: ATM Withdrawal ─────────────────────────────────────────────
    if re.search(r'\b(ATM WDL|ATM WITHDRAWAL|ATM CASH|ATM-WDL)\b', txt_u):
        return "ATM Withdrawal"

    # ── Tier 1B: Bank Charges (unambiguous fee keywords) ────────────────────
    if re.search(r'\b(SERVICE CHARGES|BANK CHARGES|STATEMENT CHARGES|CHEQUE BOOK CHARGES|ANNUAL FEE|SMS CHARGES|MARKUP|ONLINE BANKING CHARGES|PROCESSING FEE|ACCOUNT MAINTENANCE)\b', txt_u):
        return "Bank Charges"

    # ── Tier 1C: Salary (unambiguous payroll keywords) ──────────────────────
    if re.search(r'\b(SALARY|PAYROLL|WAGES|REMUNERATION|SALARY TRF|IMPS BRN SALARY|NEFT SALARY)\b', txt_u):
        return "Salary / Payroll"

    # ── Tier 1D: Investment (clear financial instrument keywords) ───────────
    if re.search(r'\b(MUTUAL FUND|SECURITIES|SIP|ZERODHA|GROWW|UPSTOX|TRADING|DEMAT|DIVIDEND|EQUITY SHARES|NIFTY|SENSEX|STOCKS|SHARES|NSE|BSE)\b', txt_u):
        return "Investment / Wealth"

    # ── Tier 1E: Healthcare (specific pharmacy/clinic brands & terms) ────────
    if re.search(r'\b(APOLLO PHARMACY|MEDPLUS|NETMEDS|PHARMEASY|TATA MED|HEALTHKART|WELLNESS FOREVER|DIAGNOSTIC|PATHOLOGY|SCAN CENTRE|HOSPITAL|NURSING HOME)\b', txt_u):
        return "Healthcare / Medical"

    # ── Tier 1F: Food & Dining (specific food brands) ───────────────────────
    if re.search(r'\b(ZOMATO|SWIGGY|BLINKIT FOOD|AMUL|ZEROX|PARLOUR|BAKERY|JUICE|RESTAURANT|CAFE|CANTEEN|DHABA|HOTEL FOOD|SNACKS|PIZZA|BURGER|DOMINOES|MCDONALDS|STARBUCKS|CHAAYOS)\b', txt_u):
        return "Food & Dining"

    # ── Tier 1G: Utility/Bills (specific board/provider names) ──────────────
    if re.search(r'\b(ELECTRICITY|MSEB|BESCOM|TNEB|TORRENT POWER|TPDDL|BSES|BEST ELECTRIC|WATER BILL|GAS BILL|PIPED GAS|BROADBAND|WIFI BILL|TELEPHONE BILL|JIOFIBER|BSNL|AIRTEL FIBER|MOBILE RECHARGE|DTH RECHARGE)\b', txt_u):
        return "Utility / Bills"

    # ── Tier 1H: Rent / Lease (specific property terms) ─────────────────────
    if re.search(r'\b(RENT PAYMENT|MONTHLY RENT|HOUSE RENT|OFFICE RENT|PROPERTY RENT|LEASE PAYMENT|LANDLORD|RENTAL DEPOSIT|ACCOMMODATION)\b', txt_u):
        return "Rent / Lease"

    # ── Tier 1I: Software / SaaS (specific tech brands) ─────────────────────
    if re.search(r'\b(SUBSCRIPTION|SAAS|SOFTWARE LICENSE|CLOUD HOSTING|AWS|GOOGLE CLOUD|AZURE|ADOBE|GITHUB|CANVA|NETFLIX|HOTSTAR|PRIME VIDEO|SPOTIFY|DROPBOX|NOTION|SLACK|ZOOM)\b', txt_u):
        return "Software / SaaS"

    # ── Tier 1J: Daily Needs / Retail (grocery & hypermarket brands) ─────────
    if re.search(r'\b(BIGBASKET|BLINKIT|ZEPTO|DUNZO|DMART|RELIANCE FRESH|JIOMART|GROFERS|SWIGGY INSTAMART|MORE RETAIL|SPAR|NATURA|HYPERCITY|WALMART|COSTCO)\b', txt_u):
        return "Daily Needs / Retail"

    # ── Tier 1K: Travel & Transport ──────────────────────────────────────────
    if re.search(r'\b(UBER|OLA CAB|RAPIDO|IRCTC|RAILWAY|AIRLINE|SPICEJET|INDIGO|AIR INDIA|MAKEMYTRIP|YATRA|OYO|GOIBIBO|PETROL|FUEL STATION|SHELL|BHARAT PETROLEUM|HP PETROL)\b', txt_u):
        return "Travel & Transport"

    # ── Tier 2: Person-name pattern detection → P2P Transfer ────────────────
    # UPI pattern: UPI-FIRSTNAME LASTNAME-VPA@BANK
    # Detect if the segment between first UPI- and the VPA looks like a person name
    # (2+ words of 2+ letters, no merchant/brand indicators)
    upi_name_m = re.match(r'UPI-([A-Z][A-Z ]{3,30})-', txt_u)
    if upi_name_m:
        candidate = upi_name_m.group(1).strip()
        # Must be only alphabetic words (no numbers, no brand-like single words)
        words = candidate.split()
        is_person = (
            len(words) >= 2
            and all(re.match(r'^[A-Z]+$', w) for w in words)
            and not re.search(r'\b(GOOGLE|AMAZON|FLIPKART|PAYTM|PHONE|HDFC|ICICI|AXIS|SBI|BANK|STORE|SHOP|MART|INDIA|PVT|LTD|CORP|ZEROX|AMUL|BLINKIT|ZOMATO|SWIGGY)\b', candidate)
        )
        if is_person:
            return "Personal / P2P Transfer"

    # Non-UPI: person name prefix patterns (RTGS/NEFT to individuals)
    if re.search(r'\b(MR|MS|MRS|DR|SHRI|SHREE|KU|MISS)\s+[A-Z]{2,}\s+[A-Z]{2,}\b', txt_u):
        return "Personal / P2P Transfer"

    # ── Tier 3: Semantic embedding fallback with strict threshold ────────────
    if emb is not None and sub_embeddings is not None and "Transaction Purpose" in sub_embeddings:
        purp_sims = {
            label: float(np.max(np.dot(sub_embeddings["Transaction Purpose"][label], emb)))
            for label in SUB_TAXONOMY["Transaction Purpose"]
        }
        best_purp, purp_score = max(purp_sims.items(), key=lambda x: x[1])
        # High threshold 0.75 to avoid false positives on ambiguous text
        return best_purp if purp_score >= 0.75 else "General Transfer"

    return "General Transfer"


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


def extract_counterparty(text: str, raw_text: str = None) -> str:
    """
    Extracts a human-readable counterparty from the raw description.
    Priority: Known entity catalog → SELF → UPI person name → segment heuristic.
    Uses raw_text when available (preserves UPI-NAME structure for accurate extraction).
    """
    # Use raw description when available for better name extraction
    src = raw_text if raw_text else text
    src_u = src.upper()

    for ent in KNOWN_ENTITIES:
        if ent in src_u:
            return ent.title()

    if re.search(r'\bSELF\b', src_u):
        return "Self"

    # UPI: format is  UPI-NAME[-gpay_or_payid]-vpa@bank-IFSC
    # Split by '-' and collect consecutive purely-alphabetic segments as the name.
    # Stop at the first segment that contains digits or '@'.
    if src_u.startswith('UPI-'):
        body = src[4:]          # Strip 'UPI-'
        segments = body.split('-')
        name_parts = []
        for seg in segments:
            seg_s = seg.strip()
            # A name segment: only letters and spaces, no '@', no leading digit, min 2 chars
            if re.match(r'^[A-Za-z][A-Za-z ]*$', seg_s) and '@' not in seg_s and len(seg_s) >= 2:
                name_parts.append(seg_s)
            elif name_parts:
                # Name collection ended — next segment broke the pattern
                break
        if name_parts:
            full_name = ' '.join(name_parts).strip()
            # Remove payment platform noise words
            full_name = re.sub(
                r'\b(GPAY|PAYTM|PHONEPE|BHIM|COLLECT REQUEST|UPI|DR|CR)\b',
                '', full_name, flags=re.IGNORECASE
            ).strip()
            if len(full_name) >= 3:
                return full_name.title()

    # Heuristic segment parsing for non-UPI narrations (NEFT/RTGS/IMPS/Cheque)
    segments = [s.strip() for s in re.split(r'[-\u2013\u2014:]', src) if s.strip()]
    for seg in reversed(segments):
        # Strip VPA handles and bank noise
        seg_clean = re.sub(r'\S*@\S+', '', seg)
        seg_clean = re.sub(
            r'\b(UPI|CR|DR|NEFT|RTGS|IMPS|CASH|BNA|ATM|WDL|CHQ|PAID|MICR|INWARD|CLEARING|SALARY|RENT|TRF|BY|TRANSFER|GPAY|PAYTM|PHONEPE|NONE|URGENT)\b',
            '', seg_clean, flags=re.IGNORECASE
        ).strip()
        if (len(seg_clean) >= 4
                and not seg_clean.strip().replace(' ', '').isdigit()
                and not re.search(
                    r'\b(BANK|CHARGES|SERVICE|STATEMENT|SUBSCRIPTION|PAYMENT|ACCTS|LIMITED|LTD|PVT|CORP)\b',
                    seg_clean, flags=re.IGNORECASE)):
            return seg_clean.strip().title()

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
        emb      = norm_embeddings[i]
        raw_txt  = raw_descriptions[i]  # Raw text for UPI-NAME pattern matching
        row_dict = df.iloc[i].to_dict() if i < len(df) else None
        modes.append(extract_transaction_mode(eval_txt, emb, sub_embeddings))
        directions.append(extract_transaction_direction(eval_txt, emb, sub_embeddings, row=row_dict))
        purposes.append(extract_transaction_purpose(eval_txt, emb, sub_embeddings, raw_text=raw_txt))
        banks.append(extract_bank_institution(eval_txt, emb, sub_embeddings))
        parties.append(extract_counterparty(eval_txt, raw_text=raw_txt))

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
        emb      = norm_embeddings[i]
        raw_txt  = raw_descriptions[i]  # Raw text for UPI-NAME pattern matching
        row_dict = df.iloc[i].to_dict() if i < len(df) else None
        modes.append(extract_transaction_mode(eval_txt, emb, sub_embeddings))
        directions.append(extract_transaction_direction(eval_txt, emb, sub_embeddings, row=row_dict))
        purposes.append(extract_transaction_purpose(eval_txt, emb, sub_embeddings, raw_text=raw_txt))
        banks.append(extract_bank_institution(eval_txt, emb, sub_embeddings))
        parties.append(extract_counterparty(eval_txt, raw_text=raw_txt))

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
