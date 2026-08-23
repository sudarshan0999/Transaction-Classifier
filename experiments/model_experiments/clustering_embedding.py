import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from embedding import generate_and_save_embeddings, load_data, resolve_target_column

def load_or_generate_embeddings(df, target_column, npy_path="embeddings.npy", input_file="extracted.xlsx"):
    """
    Loads embeddings from .npy file if exists; otherwise generates and saves them using embedding.py.
    """
    resolved_col = resolve_target_column(df, target_column)
    expected_rows = len(df)

    if os.path.exists(npy_path):
        print(f"Loading existing embeddings from '{npy_path}'...")
        embeddings = np.load(npy_path)
        if embeddings.shape[0] == expected_rows:
            print(f"Successfully loaded {embeddings.shape[0]} embeddings (Dim: {embeddings.shape[1]}).")
            return embeddings, resolved_col
        else:
            print(f"Warning: Cached embeddings rows ({embeddings.shape[0]}) mismatch data rows ({expected_rows}). Regenerating...")

    print(f"'{npy_path}' not found or row mismatch. Generating new embeddings...")
    embeddings = generate_and_save_embeddings(
        input_file=input_file,
        target_column=target_column,
        output_npy=npy_path
    )
    return embeddings, resolved_col

def perform_embedding_clustering(df, embeddings, n_clusters=4):
    """
    Performs K-Means clustering on the given embedding vectors.
    """
    print(f"\nRunning KMeans on embeddings with k={n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(embeddings)

    # Summary of cluster distribution
    print("\n--- Cluster Distribution ---")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    print(cluster_counts)

    return df, kmeans

def extract_cluster_keywords_df(df, column_name, n_clusters):
    """
    Extracts all keywords and frequencies for each cluster and returns a structured DataFrame.
    """
    records = []
    for cluster_id in range(n_clusters):
        cluster_rows = df[df['Cluster'] == cluster_id]
        cluster_size = len(cluster_rows)

        if cluster_size == 0:
            continue

        text_data = cluster_rows[column_name].astype(str).fillna('')
        cv = CountVectorizer(stop_words='english')
        try:
            word_counts = cv.fit_transform(text_data)
            sum_words = word_counts.sum(axis=0)
            words_freq = [(word, int(sum_words[0, idx])) for word, idx in cv.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

            for rank, (word, freq) in enumerate(words_freq, 1):
                records.append({
                    "Cluster": cluster_id,
                    "Total_Records_In_Cluster": cluster_size,
                    "Rank": rank,
                    "Keyword": word,
                    "Frequency": freq
                })
        except ValueError:
            pass

    return pd.DataFrame(records)

def export_results_to_excel(clustered_df, keywords_df, output_excel="clustered_embedding_output.xlsx"):
    """
    Saves both the clustered dataset and the keywords frequency summary into separate sheets in one Excel workbook.
    """
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # Sheet 1: Clustered data
        clustered_df.to_excel(writer, sheet_name="Clustered_Data", index=False)
        # Sheet 2: Cluster keywords & frequencies
        keywords_df.to_excel(writer, sheet_name="Cluster_Keywords", index=False)
        
    print(f"\nSuccessfully saved Excel with data & keyword frequencies to: '{output_excel}'")

def export_cluster_keywords_txt(df, column_name, n_clusters, output_txt="clustering_embedding.txt"):
    """
    Extracts all keywords and their frequencies for each cluster and writes them to a text file.
    """
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("       NOMIC EMBEDDING K-MEANS CLUSTERING KEYWORDS & FREQUENCY REPORT\n")
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
                words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

                for word, freq in words_freq:
                    f.write(f"{word:<40} | {freq:<10}\n")
            except ValueError:
                f.write("No distinctive keywords found.\n")

            f.write("\n\n")

    print(f"All cluster keywords with frequencies written to: '{output_txt}'")

def plot_clusters(embeddings, labels, n_clusters):
    """
    Visualizes embedding clusters in 2D using PCA along with size distribution.
    """
    print("\nGenerating cluster visualization plot...")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Nomic Embedding K-Means Clustering (k={n_clusters})", fontsize=14, fontweight='bold')

    # Panel 1: 2D PCA Scatter Plot
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
    ax1.set_title("2D PCA Projection of Embeddings", fontsize=12)
    ax1.set_xlabel("PCA Component 1", fontsize=10)
    ax1.set_ylabel("PCA Component 2", fontsize=10)
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
    plt.savefig("cluster_embedding_plot.png", dpi=300)
    print("Plot saved to 'cluster_embedding_plot.png'")
    plt.show()

def main():
    input_file = "extracted.xlsx"
    
    # -------------------------------------------------------------
    # CONFIGURATION: Set your target column and number of clusters
    # -------------------------------------------------------------
    # Set the column name (e.g. 'Description', 'Narration', 'Txn Date') or index (e.g. 3, 0)
    target_column = "Description"
    n_clusters = 2                  # Number of clusters
    npy_path = "embeddings.npy"     # Saved embeddings file
    output_excel = "clustered_embedding_output.xlsx"
    output_txt = "clustering_embedding.txt"
    # -------------------------------------------------------------

    # 1. Load Excel Data
    df, resolved_path = load_data(input_file)
    print(f"Data shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")
    
    # 2. Load embeddings from .npy (or generate once if missing)
    embeddings, resolved_col = load_or_generate_embeddings(
        df,
        target_column=target_column,
        npy_path=npy_path,
        input_file=input_file
    )

    # 3. Perform K-Means Clustering on Embeddings
    clustered_df, kmeans = perform_embedding_clustering(df, embeddings, n_clusters=n_clusters)

    # 4. Extract Keywords & Frequencies DataFrame
    keywords_df = extract_cluster_keywords_df(clustered_df, column_name=resolved_col, n_clusters=n_clusters)

    # 5. Save Clustered Output and Keywords & Frequencies to Excel (multi-sheet)
    export_results_to_excel(clustered_df, keywords_df, output_excel=output_excel)

    # 6. Write all cluster keywords & frequencies to text file
    export_cluster_keywords_txt(
        clustered_df,
        column_name=resolved_col,
        n_clusters=n_clusters,
        output_txt=output_txt
    )

    # 7. Display the cluster plot
    plot_clusters(embeddings, clustered_df['Cluster'], n_clusters)

if __name__ == "__main__":
    main()
