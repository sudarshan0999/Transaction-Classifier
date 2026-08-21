import os
import numpy as np

# =========================================================================
# Method 1: Using Nomic API (Official nomic package)
# =========================================================================
# To use the Nomic API:
# 1. Get a free API key at https://atlas.nomic.ai/
# 2. Run in terminal: nomic login <your-api-token>
#    OR set it via environment variable below:
# os.environ["NOMIC_API_KEY"] = "nk-..."

from nomic import embed

text_to_embed = "Sudarshan Ponkia"

try:
    print(f"Generating embedding for: '{text_to_embed}' using Nomic API...")
    
    # Task types: 'search_document', 'search_query', 'classification', 'clustering'
    output = embed.text(
        texts=[text_to_embed],
        model="nomic-embed-text-v1.5",
        task_type="search_document"
    )

    embedding_vector = output["embeddings"][0]
    print("\n✅ Successfully generated embedding!")
    print(f"Dimensions: {len(embedding_vector)}")
    print(f"Sample values (first 5): {embedding_vector[:5]}")
    print(f"Output shape: {np.array(output['embeddings']).shape}")

except Exception as e:
    print("\n⚠️ Nomic API Authentication required:")
    print(e)
    print("\n👉 To authenticate, run:")
    print("   nomic login <your-api-token>")
    print("   or set os.environ['NOMIC_API_KEY'] = 'your_key_here'")


# =========================================================================
# Method 2: Offline / Local embedding without API key (Optional)
# =========================================================================
# If you want to run Nomic locally without an API key, install:
# pip install sentence-transformers einops
#
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
# embeddings = model.encode(["search_document: " + text_to_embed])
# print("Local Embedding Shape:", embeddings.shape)
