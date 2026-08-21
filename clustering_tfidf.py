import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

def load_data(file_path):
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
    # 1. Exact match in DataFrame columns
    if target_col in df.columns:
        return target_col
    
    # 2. String representation match
    cols_str = {str(c).lower().strip(): c for c in df.columns}
    target_str = str(target_col).lower().strip()
    if target_str in cols_str:
        return cols_str[target_str]

    # 3. Numeric index
    if isinstance(target_col, int) and target_col < len(df.columns):
        return df.columns[target_col]
    if isinstance(target_col, str) and target_col.isdigit():
        idx = int(target_col)
        if idx < len(df.columns):
            return df.columns[idx]

    # 4. Search table content for matching column header name
    for col in df.columns:
        matching = df[df[col].astype(str).str.lower().str.strip() == target_str]
        if not matching.empty:
            print(f"Found header '{target_col}' inside table data -> Mapped to Column {col}")
            return col

    # 5. Partial / Substring search in table
    for col in df.columns:
        matching = df[df[col].astype(str).str.lower().str.contains(target_str, na=False)]
        if not matching.empty:
            print(f"Matched '{target_col}' in Column {col}")
            return col

    raise ValueError(
        f"Could not find column '{target_col}'. Available columns in Excel: {list(df.columns)}"
    )

def perform_kmeans_clustering(df, column_name, n_clusters=4):
    """
    Computes TF-IDF features on the specified column and performs K-Means clustering.
    """
    resolved_col = resolve_target_column(df, column_name)

    # Extract text from the resolved column
    text_data = df[resolved_col].astype(str).fillna('')
    
    print(f"\nVectorizing column '{resolved_col}' ({len(text_data)} rows) using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(text_data)

    print(f"Running KMeans with k={n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X)

    # Summary of cluster distribution
    print("\n--- Cluster Distribution ---")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    print(cluster_counts)

    # Top keywords per cluster
    print("\n--- Top Keywords per Cluster ---")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :6]] if len(terms) > 0 else []
        cluster_keywords[i] = ", ".join(top_terms)
        print(f"Cluster {i}: {cluster_keywords[i]}")

    return df, X, kmeans, cluster_keywords, resolved_col

def export_cluster_keywords(df, column_name, n_clusters, output_txt="clustering.txt"):
    """
    Extracts all keywords and their frequencies for each cluster and writes them into clustering.txt.
    """
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("                 K-MEANS CLUSTERING KEYWORDS & FREQUENCY REPORT\n")
        f.write("=" * 80 + "\n\n")

        for cluster_id in range(n_clusters):
            cluster_rows = df[df['Cluster'] == cluster_id]
            cluster_size = len(cluster_rows)
            
            f.write("=" * 80 + "\n")
            f.write(f"CLUSTER {cluster_id} (Total Records: {cluster_size})\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Keyword':<40} | {'Frequency':<10}\n")
            f.write("-" * 80 + "\n")

            if cluster_size == 0:
                f.write("No records in this cluster.\n\n")
                continue

            text_data = cluster_rows[column_name].astype(str).fillna('')
            cv = CountVectorizer(stop_words='english')
            try:
                word_counts = cv.fit_transform(text_data)
                sum_words = word_counts.sum(axis=0)
                words_freq = [(word, int(sum_words[0, idx])) for word, idx in cv.vocabulary_.items()]
                # Sort keywords by frequency in descending order
                words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

                for word, freq in words_freq:
                    f.write(f"{word:<40} | {freq:<10}\n")
            except ValueError:
                f.write("No distinctive keywords found.\n")

            f.write("\n\n")

    print(f"\nAll cluster keywords with frequencies written to: '{output_txt}'")

def plot_clusters(X, labels, n_clusters):
    """
    Plots the clusters and displays the window on every run.
    """
    print("\nGenerating cluster visualization plot...")
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"K-Means Clustering Analysis (k={n_clusters})", fontsize=14, fontweight='bold')

    # Panel 1: 2D Scatter plot
    scatter = ax1.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.8,
        s=60,
        edgecolors='black',
        linewidth=0.5
    )
    ax1.set_title("2D Projection of Clusters", fontsize=12)
    ax1.set_xlabel("Component 1", fontsize=10)
    ax1.set_ylabel("Component 2", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Cluster Size Distribution Bar Chart
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    bar_colors = [plt.cm.tab10(i % 10) for i in cluster_counts.index]
    
    bars = ax2.bar(
        [f"Cluster {i}" for i in cluster_counts.index],
        cluster_counts.values,
        color=bar_colors,
        edgecolor='black',
        alpha=0.85
    )
    ax2.set_title("Cluster Size Distribution", fontsize=12)
    ax2.set_ylabel("Count of Records", fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("cluster_plot.png", dpi=300)
    print("Plot saved to 'cluster_plot.png'")
    plt.show()

def main():
    input_file = "extracted.xlsx"
    
    # -------------------------------------------------------------
    # CONFIGURATION: Set your target column and number of clusters
    # -------------------------------------------------------------
    target_column = "Description"   # Can be 'Description', 3, 'Txn Date', etc.
    n_clusters = 2                  # Number of clusters
    # -------------------------------------------------------------

    # 1. Load Excel Data
    df, resolved_path = load_data(input_file)
    print(f"Data shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")
    
    # 2. Perform K-Means Clustering on the specified column
    clustered_df, X, kmeans, cluster_keywords, resolved_col = perform_kmeans_clustering(
        df, column_name=target_column, n_clusters=n_clusters
    )

    # 3. Save Clustered Output to Excel
    output_excel = "clustered_output.xlsx"
    clustered_df.to_excel(output_excel, index=False)
    print(f"\nClustered dataset successfully saved to: {output_excel}")

    # 4. Write all cluster keywords & frequencies to clustering.txt
    export_cluster_keywords(clustered_df, column_name=resolved_col, n_clusters=n_clusters, output_txt="clustering.txt")

    # 5. Display the plot
    plot_clusters(X, clustered_df['Cluster'], n_clusters)

if __name__ == "__main__":
    main()
