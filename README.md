# Bank Statement Processing & Transaction Classification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Camelot](https://img.shields.io/badge/PDF_Extractor-Camelot--py-orange.svg)](https://camelot-py.readthedocs.io/)
[![RapidOCR](https://img.shields.io/badge/OCR-RapidOCR-brightgreen.svg)](https://github.com/RapidAI/RapidOCR)
[![Nomic Embeddings](https://img.shields.io/badge/Embeddings-nomic--embed--text--v1.5-purple.svg)](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

---

## Overview

This project is a complete solution to the bank statement processing and classification assessment. The goal was to build a system that can take a bank statement PDF — whether it's a clean digital file or a scanned image — extract every transaction from it accurately, classify each transaction without using any LLM, and export the results to Excel or CSV.

The final deliverable is an interactive Streamlit web application where you can upload a PDF, see the extracted transactions, view the classification results, and download the output — all running locally with no cloud dependency.

---

## Problem Statement

The assessment required building a prototype system with the following constraints:

- Accept bank statement PDFs from multiple banks
- Support both **text-based PDFs** (digital) and **image-based PDFs** (scanned documents)
- Extract account details and transaction data: date, description, debit/credit amounts, and balance
- Classify transactions using **non-LLM approaches only** — heuristic rules or traditional ML
- Export results to Excel (`.xlsx`) and CSV (`.csv`)

The key constraint was no LLMs. This forced us to think carefully about what makes transaction classification work without GPT-style reasoning.

---

## Table of Contents

1. [How the Project Evolved](#how-the-project-evolved)
2. [Approach 1 — PDF Extraction (Why We Moved from pdfplumber to Camelot)](#approach-1--pdf-extraction)
3. [Approach 2 — OCR Pipeline for Scanned PDFs](#approach-2--ocr-pipeline-for-scanned-pdfs)
4. [Approach 3 — Description Cleaning](#approach-3--description-cleaning)
5. [Approach 4 — Unsupervised TF-IDF Clustering (EDA Phase)](#approach-4--unsupervised-tf-idf-clustering)
6. [Approach 5 — Dense Embedding Clustering (EDA Phase)](#approach-5--dense-embedding-clustering)
7. [Approach 6 — Final Classification Engine](#approach-6--final-classification-engine)
8. [Why 5 Classification Facets?](#why-5-classification-facets)
9. [The Streamlit Application](#the-streamlit-application)
10. [Repository Structure](#repository-structure)
11. [Setup & Running Locally](#setup--running-locally)
12. [Testing](#testing)
13. [Future Improvements](#future-improvements)

---

## How the Project Evolved

This project wasn't designed upfront — it evolved through experimentation. Here's a summary of the development phases mapped to git commits:

```
7bdf0e2  initial commit — sample PDFs and assessment guidelines
5f70cdd  camelot-py installation and dependency setup in try.ipynb
5c44f81  first clustering attempt using TF-IDF and K-Means
f7ca0ef  refactored clustering to TF-IDF, added cluster keyword reports
f349422  added Nomic embedding generation and PCA-based clustering
fb97fa3  automated data filtering pipeline, 4-cluster experiment
f0f0a39  description cleaning pipeline, filtered embeddings
0793d0e  5-class embedding taxonomy with exemplars and metadata
c4b1e92  Streamlit app with OCR pipeline and bank schema parsing
61200fb  semantic classification pipeline with regex + embedding hybrid
```

We started with extraction, moved to understanding the data through clustering, identified the need for cleaning, designed the taxonomy, and finally built the classifier. Each phase fed into the next.

---

## Approach 1 — PDF Extraction

### The Problem with pdfplumber

The first library we tried was `pdfplumber`. It works well for simple PDFs but struggled with bank statements for one specific reason: **multi-line cells**.

Bank statement descriptions often wrap across 2–4 lines. A single UPI transaction might read:

```
UPI-TRANSFER TO 9876543210
SALARY TRF BY - CONSUMER
PRODUCTS INDIA
```

`pdfplumber` interpreted each of those lines as a separate row. The result was a table with three times as many rows as expected, broken descriptions, and amounts appearing in completely wrong rows.

### The Problem with pytesseract

Before settling on a proper OCR engine, we also tested `pytesseract` on digital PDFs (treating them as images). This was a mistake — Tesseract has no concept of table structure. It outputs raw text in reading order, which means amount columns bleed into description text, decimal points get dropped (`1,250.00` becomes `125000`), and page headers and footers pollute the data.

### Why Camelot Works

Camelot uses **OpenCV** underneath to detect table grid lines visually. It runs in two modes:

- **Lattice mode**: Looks for ruled grid lines (explicit borders). This works perfectly for statements like `00007.pdf` and `00018.pdf` where the table has visible borders.
- **Stream mode**: Uses character whitespace to infer column boundaries. Used as a fallback for borderless tables.

Because Camelot detects the entire table as a 2D grid of cells, multi-line descriptions stay inside their correct cell. It directly returns a clean Pandas DataFrame with no row-count mismatch.

### Multi-Bank Schema Handling

The sample statements had two different column layouts:

| Schema | Format | Example Bank |
|--------|--------|-------------|
| **Schema A** | Separate `Debit` and `Credit` columns | Velocity, Prosperity, Metropolitan Banks |
| **Schema B** | Single `Amount` column with a `Cr/Dr` flag | INDUSBANK, Unity National Bank |

We built a `FORMAT_REGISTRY` that looks at the header row of each extracted table and automatically matches it to the correct schema. This means the system handles different bank formats without any manual configuration.

---

## Approach 2 — OCR Pipeline for Scanned PDFs

Two of the five sample PDFs (`00004.pdf`, `00005.pdf`) were scanned images — they had no embedded text, so Camelot couldn't extract anything from them.

For these, we built a separate pipeline:

1. **Render each page to an image** at 200–300 DPI using PyMuPDF (`fitz`)
2. **Run RapidOCR** (ONNX-based, fast, runs locally) to get word tokens with bounding box coordinates `[x_min, y_min, x_max, y_max]`
3. **Y-axis row grouping**: Group words into rows based on vertical proximity — words whose Y-centers are within a threshold `Δy` are considered the same row
4. **X-axis column assignment**: Use the header row's X-coordinates as column boundaries. Each word is assigned to the closest column slot
5. **Multi-line stitching**: If a row has a description but no amount, it gets merged into the preceding row

We chose RapidOCR over Tesseract because it's significantly faster and more accurate on printed text, and critically it returns structured bounding boxes rather than a flat string — which is essential for reconstructing table geometry.

The system auto-detects whether a PDF is digital or scanned by checking the character count of the first page. If it's below 80 characters, it routes to the OCR pipeline; otherwise, it uses Camelot.

---

## Approach 3 — Description Cleaning

Once we had the raw descriptions extracted, the next problem was noise. A raw description looks like this:

```
NEFT-CNRBR946608852502-HDFC0004171-RAMESH VORA-/NONE
```

For classification, the meaningful part is `RAMESH VORA` and `NEFT`. Everything else — the UTR number (`CNRBR946608852502`), the IFSC code (`HDFC0004171`), the `/NONE` tag — is noise that would confuse any classifier.

We built a dedicated cleaning function ([`clean_description()`](file:///c:/Users/sudar/OneDrive/Desktop/New%20approach/classify_pipeline.py#L142-L177)) with these steps:

1. **Fix OCR-split words**: Sometimes a name like `RAMESHWAR` gets split as `RAMES` + newline + `HWAR` by OCR. We patch specific known prefixes.
2. **Remove boilerplate bank tags**: `/NONE`, `/URGENT`, `SENT-TRANSFER FROM`, `UPI-TRANSFER TO` followed by phone numbers
3. **Strip IFSC codes**: Regex pattern `[A-Z]{4}0[A-Z0-9]{6}`
4. **Strip all numeric reference IDs**: UTR numbers, phone numbers, ATM IDs — any alphanumeric token containing digits
5. **Normalize punctuation and whitespace**

After cleaning, the same description becomes:

```
NEFT - RAMESH VORA
```

That's actually classifiable.

---

## Approach 4 — Unsupervised TF-IDF Clustering

Before designing the classifier, we needed to understand the data. The goal here wasn't to build a production classifier — it was to answer the question: **do bank descriptions have any natural structure we can exploit?**

We used TF-IDF vectorization (`max_features=500`, English stop words) and K-Means clustering, testing K=2, 4, and 8 on the extracted descriptions.

**What we found:**

At K=4, the clusters were clean and interpretable:

- **Cluster 0** (117 records, 65%): Electronic transfers — UPI, NEFT, IMPS, RTGS, salary payments
- **Cluster 1** (38 records, 21%): Cheque clearing and interbank settlements — MICR, Citi Bank DEL ACCTS, CHQ PAID
- **Cluster 2** (19 records, 11%): Cash and ATM — CASH BNA, ATM WDL, self withdrawal
- **Cluster 3** (6 records, 3%): Repeated table headers ("Description", "Txn Date") — pure noise from multi-page PDF extraction

This confirmed two things:
1. The descriptions have genuine semantic structure
2. Multi-page extraction was pulling in header rows as data rows — which led us to build the cleaning pipeline

**Why we didn't stop here:** TF-IDF clustering produces anonymous cluster IDs (0, 1, 2, 3). There's no guarantee the IDs are consistent across different runs or different datasets. And a single cluster number per transaction loses most of the information — the same transaction can simultaneously have a payment mode, a purpose, a direction, and a counterparty.

The cluster visualizations from this phase are in the [`images/`](file:///c:/Users/sudar/OneDrive/Desktop/New%20approach/images) directory — 2D scatter plots (via TruncatedSVD) across K=2, 4, and 8.

---

## Approach 5 — Dense Embedding Clustering

TF-IDF treats each word as an independent feature. "Cash Withdrawal" and "ATM WDL" would land in different parts of the vector space because they share no words, even though they mean the same thing.

We tested `nomic-ai/nomic-embed-text-v1.5` (768 dimensions, 8192-token context) as an alternative. This model converts a description into a dense vector that captures semantic meaning, so synonyms and abbreviations land near each other in vector space.

We ran PCA to project the 768D vectors down to 2D and plotted the clusters. The results were visually cleaner — descriptions that mean the same thing clustered tightly regardless of how they were written.

**Key observation:** At K=8, the dense embedding clusters started mapping cleanly onto recognizable banking categories: salary transfers, ATM cash, cheque clearing, utility payments, etc. This gave us the confidence to move from unsupervised clustering to a supervised prototype approach for the final classifier.

Both sets of cluster plots are in [`images/`](file:///c:/Users/sudar/OneDrive/Desktop/New%20approach/images).

---

## Approach 6 — Final Classification Engine

The clustering experiments told us what categories exist. The final step was turning that into a production classifier.

### Why We Don't Use a Single Category

A raw transaction description like:

```
UPI CR - HDFC BANK - SALARY TRF - CONSUMER PRODUCTS INDIA
```

contains at least four independent pieces of information:
- Payment mode: **UPI**
- Cash flow direction: **Credit**
- Business purpose: **Salary / Payroll**
- Counterparty: **Consumer Products India**

Assigning a single label like "UPI" or "Salary" means throwing away the rest. For any real financial use case — reconciliation, auditing, expense categorization — you need all of it.

### The 5 Classification Facets

We classify every transaction across 5 independent axes:

| Facet | What it captures | Example values |
|-------|-----------------|----------------|
| **Transaction Mode** | Payment channel / rail | UPI, NEFT, RTGS, IMPS, ATM, Cheque, NetBanking |
| **Transaction Direction** | Cash flow direction | Debit, Credit |
| **Transaction Purpose** | Business / economic intent | Salary, Rent, Utility Bills, Healthcare, Investment |
| **Bank / Institution** | Intermediary bank involved | HDFC, ICICI, SBI, Citi, Axis |
| **Party / Counterparty** | Who sent or received the money | Individual name or company name |

### Why These 5 and Not More?

We considered other facets (e.g., transaction frequency, amount range, geography) but ruled them out because:
- The description text alone doesn't reliably encode those dimensions
- Adding more facets without reliable signal increases noise, not accuracy

These 5 were chosen because every single transaction description in the dataset could be placed into each one with high confidence, and each facet is genuinely independent — knowing the mode tells you nothing about the purpose.

### How the Classification Works

For each facet, we use a **two-pass approach**:

**Pass 1 — Regex rules (fast, deterministic):**
We check for high-confidence keywords first. If the description contains `\bUPI\b`, the mode is UPI — no need for any computation. These rules cover the majority of transactions instantly.

```python
if re.search(r'\bUPI\b', txt_u):   return "UPI"
elif re.search(r'\bNEFT\b', txt_u): return "NEFT"
elif re.search(r'\bRTGS\b', txt_u): return "RTGS"
# ... and so on
```

**Pass 2 — Semantic similarity (for ambiguous cases):**
When rules don't fire, we embed the cleaned description using `nomic-embed-text-v1.5` and compare it (via cosine similarity) against pre-computed exemplar embeddings for each category:

```
Score(category) = max_j ( description_vector · exemplar_vector_j )
```

Each category has 2–4 hand-written exemplar sentences that describe what that category looks like. These are embedded once at startup and cached to disk. Classification at inference time is just a dot product.

**Pass 3 — Fallback:**
If the similarity score is below the threshold (e.g., 0.62 for purpose), we assign `"General Transfer"` or `"Other"` rather than making a low-confidence guess.

This hybrid approach gives deterministic results for known patterns and graceful semantic generalization for edge cases.

---

## Why 5 Classification Facets?

This came directly from studying the cluster results. At K=8 with dense embeddings, the emerging clusters weren't cleanly orthogonal "transaction types" — they were mixtures. A cluster would contain both UPI and NEFT transfers to the same type of counterparty. Another would contain salary transfers across multiple payment modes.

This told us the description space is **multi-dimensional**, not flat. The natural dimensions are: *how was the money moved, in which direction, for what purpose, through which bank, and to/from whom*. Those are exactly the 5 facets we built.

From an accounting perspective, these are also the exact dimensions that analysts use when categorizing transactions for P&L reports, bank reconciliation, and compliance checks. The classifier produces output that is directly usable by a finance team without any post-processing.

---

## The Streamlit Application

The application ([`app.py`](file:///c:/Users/sudar/OneDrive/Desktop/New%20approach/app.py)) ties everything together:

- **Upload**: Accepts single or multiple PDF files
- **Auto-detection**: Checks character density to route each PDF to the correct extractor (Camelot or RapidOCR)
- **Extraction**: Parses the transaction table with multi-bank schema support
- **Classification**: Runs the full 5-facet pipeline on every row
- **Dashboard**: Shows total transactions, total debits, total credits, net cash flow, and primary transaction channel as live metrics
- **Data grid**: Lets you browse and inspect individual transactions with all 5 classification labels
- **Export**: Download as formatted Excel (`.xlsx`) or CSV (`.csv`)

The ML model and embeddings are loaded once at startup and cached in the session (`@st.cache_resource`), so classification on a 200-row statement takes under 2 seconds.

---

## Repository Structure

```
.
├── app.py                          # Streamlit application (main entry point)
├── classify_pipeline.py            # 5-facet classification engine + text cleaning
├── create_particular_embedding.py  # Builds and caches the exemplar embeddings
├── embedding.py                    # Nomic embedding generation utility
├── clustering_tfidf.py             # TF-IDF + K-Means clustering (EDA phase)
├── clustering_embedding.py         # Dense embedding + K-Means clustering (EDA phase)
├── data_filter.py                  # Raw data filtering and header noise removal
├── data_filtering_1description.py  # Description-only cleaning pipeline
├── requirement.txt                 # Python dependencies
│
├── data/
│   ├── 00004.pdf                   # Scanned bank statement (image-based)
│   ├── 00005.pdf                   # Scanned bank statement (image-based)
│   ├── 00007.pdf                   # Digital statement — Schema A (Debit/Credit columns)
│   ├── 00009.pdf                   # Digital statement — Schema A
│   ├── 00018.pdf                   # Digital statement — Schema B (Cr/Dr flag)
│   └── Assessment Document.pdf     # Original problem statement
│
├── embedding_class/                # Pre-computed exemplar embeddings (cached)
│   ├── class_embeddings_bundle.npz
│   ├── class_metadata.json
│   └── sub_taxonomy_embeddings.npz
│
├── images/                         # Cluster plots from EDA (K=2, 4, 8 — TF-IDF and Nomic)
│
├── kmeans/                         # Text reports from clustering experiments
│
├── output/
│   ├── classified_statements/      # Final classified Excel exports
│   └── embeddings/                 # Per-statement embedding arrays (.npy)
│
└── scratch/                        # Test and verification scripts
    ├── test_pdf_to_classified.py
    └── test_subclass_classifier.py
```

---

## Setup & Running Locally

### Prerequisites

- Python 3.10 or higher (3.11 recommended)
- **Ghostscript** — required by Camelot for PDF parsing
  - Windows: Install from [ghostscript.com](https://www.ghostscript.com/download/gsdnld.html) and add the `bin/` folder to your system PATH
  - Linux: `sudo apt-get install ghostscript libgl1-mesa-glx`

### Install & Run

```bash
# Clone the repo
git clone https://github.com/sudarshan0999/Transaction-Classifier.git
cd Transaction-Classifier

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirement.txt

# Launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Running Individual Scripts

```bash
# TF-IDF clustering analysis (EDA)
python clustering_tfidf.py

# Dense embedding clustering (EDA)
python clustering_embedding.py

# Rebuild exemplar embedding cache
python create_particular_embedding.py

# End-to-end pipeline test
python scratch/test_pdf_to_classified.py
```

---

## Testing

We verified the pipeline with automated tests in `scratch/`:

```bash
python scratch/test_pdf_to_classified.py
```

Checks performed:
- Extraction row count matches the source PDF
- All 5 classification columns are present and non-empty
- Embedding dimensions are `(N, 768)` with unit L2 norm
- No internal confidence scores leak into the output Excel
- Both Schema A and Schema B layouts parse correctly

---

## Approach Comparison

| | TF-IDF + K-Means | Nomic Embeddings + K-Means | Final Hybrid Classifier |
|-|-----------------|---------------------------|------------------------|
| Supervision | None | None | Supervised prototypes + rules |
| Output | Cluster IDs (unstable) | Cluster IDs (unstable) | Named labels (consistent) |
| Cross-run consistency | No | No | Yes |
| Handles synonyms | No | Yes | Yes |
| Latency | ~2ms/txn | ~25ms/txn | <5ms/txn (cached) |
| Banking domain awareness | None | General semantic | Full banking taxonomy |
| Used for | EDA only | EDA only | Production |

---

## Future Improvements

- **Feedback loop**: Let analysts correct classifications in the UI and use corrections to update exemplars
- **Multi-currency support**: Handle USD, EUR, GBP, AED alongside INR
- **REST API**: Expose a `/classify` endpoint for integration with ERP and accounting systems
- **Docker image**: Single-command deployment bundling all dependencies including Ghostscript

---

## Acknowledgments

Built using [Streamlit](https://streamlit.io/), [Camelot-py](https://camelot-py.readthedocs.io/), [RapidOCR](https://github.com/RapidAI/RapidOCR), [Sentence-Transformers](https://sbert.net/), and [Nomic AI](https://nomic.ai/).
