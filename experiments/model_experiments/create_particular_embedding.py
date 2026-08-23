import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 1. 5-CLASS DOMAIN TAXONOMY & HIGH-PRECISION EXEMPLARS DEFINITION
# ==============================================================================
CLASS_DEFINITIONS = {
    "Transaction Mode": {
        "id": 1,
        "folder_key": "1_transaction_mode",
        "description": "Payment method, channel, or transaction rail",
        "exemplars": [
            "UPI payment Unified Payments Interface instant mobile remittance",
            "NEFT National Electronic Funds Transfer interbank payment electronic settlement",
            "RTGS Real Time Gross Settlement fund transfer high value payment",
            "IMPS Immediate Payment Service 24x7 instant remittance branch transfer",
            "CASH BNA Branch Cash Deposit Automated Machine transaction",
            "ATM WDL ATM Cash automated teller machine cash withdrawal",
            "Chq Paid Cheque clearing MICR Inward Clearing cheque settlement",
            "By Clg Clearing Cheque clearing house transaction",
            "Online Banking NetBanking digital internet transaction fee charges",
            "Service Charges statement charges cheque book charges fee"
        ]
    },
    "Transaction Direction": {
        "id": 2,
        "folder_key": "2_transaction_direction",
        "description": "Cash flow movement indicating debit outflow or credit inflow",
        "exemplars": [
            "Dr Debit outgoing payment money debited debit transfer",
            "Cr Credit incoming payment money credited credit received",
            "UPI DR NEFT Dr RTGS Dr IMPS Dr Debit payment transfer out",
            "UPI CR NEFT Cr RTGS Cr IMPS Cr Credit received transfer in",
            "Cash Withdrawal ATM WDL funds withdrawn debit outflow payout",
            "Cash Deposit BNA Deposit funds deposited credit inflow deposit",
            "Chq Paid Cheque Paid amount debited outward clearing",
            "MICR Inward Clearing By Clg credit incoming cheque clearing"
        ]
    },
    "Transaction Purpose": {
        "id": 3,
        "folder_key": "3_transaction_purpose",
        "description": "Underlying purpose, reason, or business category of the transaction",
        "exemplars": [
            "SALARY TRF Salary transfer monthly employee payroll remuneration staff pay",
            "RENT PAYMENT MONTHLY RENT house office rent lease payment",
            "SOFTWARE SUBSCRIPTION tech cloud software license SaaS billing",
            "UTILITY PAYMENT electricity water gas internet telephone bill",
            "Service Charges Online Banking Charges internet fee maintenance fee",
            "Service Charges Cheque Book Charges cheque book issue fee",
            "Service Charges Statement Charges bank statement request fee",
            "Service Charges Foreign Currency Markup FX international transaction fee",
            "INVESTMENT SERVICES mutual funds securities wealth management trading",
            "HEALTHCARE Medical Solutions remedies pharma wellness medicines health clinic",
            "DAILY NEEDS Retail Brands consumer essentials clothing trading products goods"
        ]
    },
    "Bank / Institution": {
        "id": 4,
        "folder_key": "4_bank_institution",
        "description": "Commercial bank, financial institution, branch, or clearing authority",
        "exemplars": [
            "HDFC BANK LTD commercial banking institution financial corporation",
            "CITI BANK N.A. CIT foreign bank financial institution",
            "ICICI Bank Ltd commercial banking corporation",
            "State Bank of India SBI nationalized bank",
            "Axis Bank Kotak Mahindra Bank Bank of Baroda",
            "BNA Branch Network Automated Cash Machine bank branch",
            "DEL ACCTS Clearing House MICR banking system institution"
        ]
    },
    "Party / Counterparty": {
        "id": 5,
        "folder_key": "5_party_counterparty",
        "description": "Beneficiary, payer, individual person, or corporate business counterparty",
        "exemplars": [
            # Individual customer names
            "RAMESH VORA NISHA JHAVERI KETAN MODI KAVITA AMIN SEJAL DAVE NILESH DOSHI",
            "NEHA PAREKH DHARA KOTHARI BHAVESH MODI RITU MEHTA RIDDHI BHATT ANJALI SHUKLA",
            "POOJA BHATT RAKESH VORA PRIYA VORA DINESH PAREKH KINJAL AMIN FORAM VYAS",
            "MANISH VYAS HIRAL PAREKH MANISH TRIVEDI SHRUTI MODI ASHISH THAKKAR MEHUL DAVE",
            "SURESH PAREKH SURESH DESAI MEHUL PATEL FORAM AMIN ANJALI DAVE RITU PATEL",
            "HITESH KOTHARI RAJESH VORA KAVITA GANDHI PRIYA MODI KAVITA DESAI ASHISH KOTHARI",
            "PRIYA MEHTA HIRAL JOSHI MEHUL PANDYA JAYESH VORA BHAVESH THAKKAR SEJAL PATEL",
            "KAVITA KOTHARI SHRUTI MEHTA ASHISH AMIN NISHA MEHTA PARESH PARIKH ANJALI PATEL",
            # Corporate and merchant entities
            "CONSUMER PRODUCTS INDIA DAILY NEEDS LIMITED WEBSTREAM LIMITED RETAIL BRANDS CORP",
            "COURIER NETWORKS INDIA COMMERCIAL PROPERTIES INDIA CONSUMER ESSENTIALS INDIA",
            "PROPERTY DEVELOPERS CORP FINANCIAL PRODUCTS LTD MEDICAL SOLUTIONS INDIA",
            "TRADING HOUSE LTD URBAN CONSTRUCTIONS NETFORGE TECHNOLOGIES PREMIUM PROPERTIES LTD",
            "FOUNDATION PROJECTS LTD DATABRIDGE INFOTECH FASHION TEXTILES LTD REMEDIES HEALTHCARE",
            "WELLNESS BRANDS INDIA HEALTHCARE PRODUCTS LTD CYBERCRAFT SYSTEMS ESTATE BUILDERS INDIA",
            "HEALTHPLUS PHARMA LTD GATEWAY PROJECTS LTD URBAN ESTATES INDIA MOTOR INDUSTRIES LTD",
            "SKYLINE BUILDERS MINDWAVE CONSULTING THERMAL SYSTEMS LTD MEDCARE LABORATORIES",
            "CLOUDTECH SERVICES VITALITY MEDICINES LTD CARGO SOLUTIONS LTD CLOTHING INDUSTRIES LTD",
            "CITYSCAPE DEVELOPERS MECHANICAL SOLUTIONS LTD MARKET DISTRIBUTORS CORP",
            "RESIDENTIAL PROJECTS LTD POWER SYSTEMS CORP LOAN FINANCE CORP COMMERCE SOLUTIONS LTD",
            "BUILDING SOLUTIONS LTD WELLNESS LABS INDIA"
        ]
    }
}

# ==============================================================================
# 2. EMBEDDING GENERATION AND STORAGE IN embedding_class/
# ==============================================================================
def create_and_save_class_embeddings(
    output_dir="embedding_class",
    model_name="nomic-ai/nomic-embed-text-v1.5"
):
    """
    Generates high-dimensional embeddings for each of the 5 classes using Nomic Embed.
    Saves individual class embeddings, prototype centroids, bundle, and metadata in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("STEP 1: GENERATING AND SAVING 5-CLASS EMBEDDINGS")
    print("=" * 80)
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    class_prototypes = []
    class_names = []
    bundle_dict = {}
    metadata = {
        "model_name": model_name,
        "classes": {}
    }

    for class_name, class_info in CLASS_DEFINITIONS.items():
        cid = class_info["id"]
        folder_key = class_info["folder_key"]
        exemplars = class_info["exemplars"]
        
        # Nomic standard search_document prefix
        prefixed_exemplars = [f"search_document: {text}" for text in exemplars]
        
        print(f"\n[{cid}/5] Encoding Class: '{class_name}' ({len(exemplars)} exemplars)...")
        exemplar_embeddings = model.encode(prefixed_exemplars, show_progress_bar=False)
        exemplar_embeddings = np.array(exemplar_embeddings, dtype=np.float32)

        # L2-normalize individual exemplar embeddings
        norms = np.linalg.norm(exemplar_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_exemplar_embeddings = exemplar_embeddings / norms

        # Compute pooled class prototype (centroid) and normalize
        centroid = np.mean(norm_exemplar_embeddings, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        class_prototypes.append(centroid)
        class_names.append(class_name)

        # Save individual class .npy file
        npy_filename = f"{folder_key}.npy"
        npy_path = os.path.join(output_dir, npy_filename)
        np.save(npy_path, norm_exemplar_embeddings)
        print(f"  -> Saved exemplar embeddings: '{npy_path}' (Shape: {norm_exemplar_embeddings.shape})")

        # Save individual centroid .npy file
        centroid_filename = f"{folder_key}_centroid.npy"
        centroid_path = os.path.join(output_dir, centroid_filename)
        np.save(centroid_path, centroid)

        # Add to bundle and metadata
        bundle_dict[f"{folder_key}_exemplars"] = norm_exemplar_embeddings
        bundle_dict[f"{folder_key}_centroid"] = centroid
        
        metadata["classes"][class_name] = {
            "id": cid,
            "folder_key": folder_key,
            "description": class_info["description"],
            "exemplar_count": len(exemplars),
            "exemplars": exemplars,
            "files": {
                "exemplars": npy_filename,
                "centroid": centroid_filename
            }
        }

    # Save all class prototype centroids as a single matrix (5, 768)
    class_prototypes = np.array(class_prototypes, dtype=np.float32)
    prototypes_path = os.path.join(output_dir, "class_prototypes.npy")
    np.save(prototypes_path, class_prototypes)
    print(f"\n-> Saved all 5 class prototypes: '{prototypes_path}' (Shape: {class_prototypes.shape})")

    # Save complete .npz bundle
    bundle_dict["class_prototypes"] = class_prototypes
    bundle_dict["class_names"] = np.array(class_names)
    bundle_path = os.path.join(output_dir, "class_embeddings_bundle.npz")
    np.savez_compressed(bundle_path, **bundle_dict)
    print(f"-> Saved compressed bundle: '{bundle_path}'")

    # Save metadata JSON
    metadata_path = os.path.join(output_dir, "class_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"-> Saved metadata definition: '{metadata_path}'")

    print("\nAll 5 class embeddings successfully generated and saved into 'embedding_class/' folder!")
    return model, class_prototypes, metadata


# ==============================================================================
# 3. LOADING EMBEDDINGS AND DATASETS
# ==============================================================================
def load_class_embeddings(embedding_dir="embedding_class"):
    """
    Loads saved class embeddings and metadata from the embedding_class folder.
    """
    metadata_path = os.path.join(embedding_dir, "class_metadata.json")
    prototypes_path = os.path.join(embedding_dir, "class_prototypes.npy")

    if not os.path.exists(metadata_path) or not os.path.exists(prototypes_path):
        raise FileNotFoundError(
            f"Class embeddings not found in '{embedding_dir}'. Run create_and_save_class_embeddings() first."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_prototypes = np.load(prototypes_path)
    
    # Load exemplar matrices per class
    class_exemplars = {}
    for class_name, info in metadata["classes"].items():
        ex_file = os.path.join(embedding_dir, info["files"]["exemplars"])
        class_exemplars[class_name] = np.load(ex_file)

    return class_prototypes, class_exemplars, metadata

def load_description_data(
    excel_path="filtered_only description.xlsx",
    npy_path="description_only_embeddings.npy",
    model=None
):
    """
    Loads descriptions from Excel and their corresponding embeddings from .npy.
    """
    if not os.path.exists(excel_path):
        fallback = "filtered_description.xlsx"
        if os.path.exists(fallback):
            excel_path = fallback
        else:
            excel_path = "extracted_full.xlsx"

    print(f"\nLoading description dataset from: '{excel_path}'...")
    df = pd.read_excel(excel_path)
    desc_col = "Description" if "Description" in df.columns else df.columns[0]
    descriptions = df[desc_col].astype(str).tolist()
    print(f"Total descriptions loaded: {len(descriptions)}")

    if os.path.exists(npy_path):
        print(f"Loading existing embeddings from: '{npy_path}'...")
        embeddings = np.load(npy_path)
        if len(embeddings) != len(descriptions):
            print(f"Warning: Count mismatch (Embeddings: {len(embeddings)}, Texts: {len(descriptions)}).")
    else:
        print(f"'{npy_path}' not found. Generating fresh description embeddings...")
        if model is None:
            model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        prefixed = [f"search_document: {t}" for t in descriptions]
        embeddings = model.encode(prefixed, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        np.save(npy_path, embeddings)

    # Normalize description embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_embeddings = embeddings / norms

    return df, descriptions, norm_embeddings


# ==============================================================================
# 4. SIMILARITY CALCULATION & THRESHOLD-BASED CLASSIFICATION
# ==============================================================================
def compute_similarity_matrix(description_embeddings, class_exemplars, mode="max_pool"):
    """
    Computes similarity scores for each description against each of the 5 classes.
    Modes:
      - 'max_pool': Maximum cosine similarity across all exemplars in a class (Best for multi-faceted classes)
      - 'top_k_mean': Average of top-k closest exemplars in each class
    """
    class_names = list(class_exemplars.keys())
    num_descriptions = len(description_embeddings)
    num_classes = len(class_names)
    
    sim_matrix = np.zeros((num_descriptions, num_classes), dtype=np.float32)

    for c_idx, class_name in enumerate(class_names):
        exemplar_matrix = class_exemplars[class_name] # (N_exemplars, Dim)
        # Cosine similarity matrix between all descriptions and all exemplars of this class
        pair_sims = np.dot(description_embeddings, exemplar_matrix.T)
        
        if mode == "max_pool":
            sim_matrix[:, c_idx] = np.max(pair_sims, axis=1)
        elif mode == "top_k_mean":
            k = min(3, pair_sims.shape[1])
            top_k = np.sort(pair_sims, axis=1)[:, -k:]
            sim_matrix[:, c_idx] = np.mean(top_k, axis=1)
        else:
            sim_matrix[:, c_idx] = np.mean(pair_sims, axis=1)

    return sim_matrix, class_names


# ==============================================================================
# 5. POPULATE EXCEL WITH YES/NO AND SCORE COLUMNS
# ==============================================================================
def populate_excel_with_class_scores(
    excel_path="filtered_only description.xlsx",
    sim_matrix=None,
    class_names=None,
    threshold=0.58,
    per_class_thresholds=None
):
    """
    Adds Yes/No and Score columns for all 5 classes into the Excel file and saves it.
    """
    df = pd.read_excel(excel_path)
    desc_col = "Description" if "Description" in df.columns else df.columns[0]
    
    # Retain Description as the primary first column
    updated_df = pd.DataFrame()
    updated_df[desc_col] = df[desc_col]

    if per_class_thresholds is None:
        threshold_dict = {name: threshold for name in class_names}
    else:
        threshold_dict = {name: per_class_thresholds.get(name, threshold) for name in class_names}

    for c_idx, class_name in enumerate(class_names):
        scores = np.round(sim_matrix[:, c_idx], 4)
        th = threshold_dict[class_name]
        yes_no_labels = ["Yes" if s >= th else "No" for s in scores]

        # Add both Yes/No and Score columns for each of the 5 classes
        updated_df[f"{class_name} (Yes/No)"] = yes_no_labels
        updated_df[f"{class_name} (Score)"] = scores

    # Save to Excel
    print(f"\nUpdating Excel file: '{excel_path}' with 5 class columns...")
    try:
        updated_df.to_excel(excel_path, index=False)
        print(f"  [Success] Saved directly to '{excel_path}'")
    except PermissionError:
        alt_excel = os.path.splitext(excel_path)[0] + "_with_classes.xlsx"
        print(f"  [Notice] '{excel_path}' is open in another program. Saved to '{alt_excel}' instead.")
        updated_df.to_excel(alt_excel, index=False)

    return updated_df


# ==============================================================================
# 6. THRESHOLD SENSITIVITY & ON-DEMAND PREDICTION
# ==============================================================================
def evaluate_threshold_sensitivity(
    sim_matrix,
    class_names,
    thresholds=(0.45, 0.50, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80)
):
    """
    Evaluates and prints match distribution across various threshold values.
    """
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY SWEEP ANALYSIS")
    print("=" * 80)
    print(f"{'Threshold':<10} | {'Classified':<12} | {'Unclassified':<14} | {'Avg Matches/Desc':<18} | {'Per-Class Match Counts'}")
    print("-" * 80)

    total_items = len(sim_matrix)

    for th in thresholds:
        match_mask = (sim_matrix >= th)
        matches_per_row = np.sum(match_mask, axis=1)
        classified_count = np.sum(matches_per_row > 0)
        unclassified_count = total_items - classified_count
        avg_matches = np.mean(matches_per_row)
        
        per_class_counts = [f"{class_names[c][:7]}: {np.sum(match_mask[:, c])}" for c in range(len(class_names))]
        per_class_summary = ", ".join(per_class_counts)

        classified_pct = (classified_count / total_items) * 100
        print(f"{th:<10.2f} | {classified_count:>4} ({classified_pct:>4.1f}%) | {unclassified_count:>5} ({(100 - classified_pct):>4.1f}%) | {avg_matches:>10.2f}        | {per_class_summary}")

    print("=" * 80)


def classify_text(text, model, class_exemplars, threshold=0.58):
    """
    Classifies a single description string against the 5 classes using the given threshold.
    """
    prefixed_text = f"search_document: {text}"
    text_emb = model.encode([prefixed_text], show_progress_bar=False)[0]
    norm = np.linalg.norm(text_emb)
    if norm > 0:
        text_emb = text_emb / norm
    
    scores = {}
    matched = []
    for class_name, ex_matrix in class_exemplars.items():
        sims = np.dot(ex_matrix, text_emb)
        max_sim = float(np.max(sims))
        scores[class_name] = round(max_sim, 4)
        if max_sim >= threshold:
            matched.append((class_name, round(max_sim, 4)))

    top_class = max(scores, key=scores.get)
    return {
        "text": text,
        "threshold": threshold,
        "top_class": top_class,
        "top_score": scores[top_class],
        "matched_classes": matched,
        "all_scores": scores
    }


# ==============================================================================
# 7. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    # --------------------------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------------------------
    input_excel = "filtered_only description.xlsx"
    description_npy = "description_only_embeddings.npy"
    embedding_folder = "embedding_class"
    
    # Base similarity threshold for Yes/No detection (Score >= threshold -> Yes)
    # 0.58 provides clean separation between present and absent classes
    SIMILARITY_THRESHOLD = 0.58
    
    # Optional: Fine-tuned per-class thresholds
    PER_CLASS_THRESHOLDS = {
        "Transaction Mode": 0.58,
        "Transaction Direction": 0.58,
        "Transaction Purpose": 0.55,
        "Bank / Institution": 0.58,
        "Party / Counterparty": 0.58
    }
    # --------------------------------------------------------------------------

    # Step 1: Generate & Save 5-Class Embeddings into embedding_class/
    model, class_prototypes, metadata = create_and_save_class_embeddings(
        output_dir=embedding_folder,
        model_name="nomic-ai/nomic-embed-text-v1.5"
    )

    # Step 2: Load Class Embeddings
    class_prototypes, class_exemplars, metadata = load_class_embeddings(embedding_dir=embedding_folder)

    # Step 3: Load Description Dataset & Pre-computed Embeddings
    raw_df, descriptions, desc_embeddings = load_description_data(
        excel_path=input_excel,
        npy_path=description_npy,
        model=model
    )

    # Step 4: Compute Cosine Similarities across 5 Classes
    print("\n" + "=" * 80)
    print("STEP 2: COMPUTING SIMILARITIES & THRESHOLD MATCHING")
    print("=" * 80)
    sim_matrix, class_names = compute_similarity_matrix(desc_embeddings, class_exemplars, mode="max_pool")

    # Step 5: Evaluate Threshold Sensitivity Sweep
    evaluate_threshold_sensitivity(sim_matrix, class_names)

    # Step 6: Populate Excel file with Yes/No and Score columns for all 5 classes
    updated_df = populate_excel_with_class_scores(
        excel_path=input_excel,
        sim_matrix=sim_matrix,
        class_names=class_names,
        threshold=SIMILARITY_THRESHOLD,
        per_class_thresholds=PER_CLASS_THRESHOLDS
    )

    # Step 7: Print Preview of Populated Excel
    print("\n" + "=" * 110)
    print("SAMPLE PREVIEW OF UPDATED EXCEL DATASET (FIRST 15 ROWS)")
    print("=" * 110)
    preview_cols = [
        "Description",
        "Transaction Mode (Yes/No)", "Transaction Mode (Score)",
        "Transaction Direction (Yes/No)", "Transaction Direction (Score)",
        "Transaction Purpose (Yes/No)", "Transaction Purpose (Score)",
        "Bank / Institution (Yes/No)", "Bank / Institution (Score)",
        "Party / Counterparty (Yes/No)", "Party / Counterparty (Score)"
    ]
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)
    print(updated_df[preview_cols].head(15).to_string())
    print("=" * 110)

    # Step 8: Demo on-demand classification
    print("\n" + "=" * 80)
    print("DEMO: REAL-TIME ON-THE-FLY TEXT CLASSIFICATION")
    print("=" * 80)
    sample_queries = [
        "UPI CR DHARA KOTHARI",
        "Chq Paid - MICR Inward Clearing - FORAM JOSHI - HDFC BANK LTD",
        "IMPS BRN SALARY TRF BY - CONSUMER PRODUCTS INDIA",
        "Service Charges - Online Banking Charges",
        "NEFT Dr - RENT PAYMENT - MONTHLY RENT"
    ]
    for sq in sample_queries:
        res = classify_text(sq, model, class_exemplars, threshold=SIMILARITY_THRESHOLD)
        print(f"\nQuery: '{sq}'")
        print(f"  Top Class: {res['top_class']} (Score: {res['top_score']})")
        print(f"  Matched Classes (>= {SIMILARITY_THRESHOLD}): {res['matched_classes']}")
        print(f"  All Scores: {res['all_scores']}")

    print("\n" + "=" * 80)
    print(f"SUCCESS: '{input_excel}' has been updated with all 5 classes (Yes/No and Scores)!")
    print("=" * 80)


if __name__ == "__main__":
    main()
