import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the data
small_list_df = pd.read_csv("small_careers_list.csv", header=None, names=["Career"])
large_list_df = pd.read_csv("large_careers_list.csv", header=None, names=["Career"])

small_careers = small_list_df["Career"].tolist()
large_careers = large_list_df["Career"].tolist()

# Load the pre-trained Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode the career names into vectors
small_embeddings = model.encode(small_careers, convert_to_tensor=True)
large_embeddings = model.encode(large_careers, convert_to_tensor=True)

# Store the results
results = []

# For each career in the small list, find the most semantically similar careers in the large list
for i, small_career in enumerate(small_careers):
    # Compute cosine similarity between this career and all careers in the large list
    similarities = util.pytorch_cos_sim(small_embeddings[i], large_embeddings)
    
    # Get the indices of top matches, let's say top 5 matches
    top_matches = np.argpartition(similarities[0], -5)[-5:]
    
    # Get the corresponding career names from large list
    matched_careers = [large_careers[idx] for idx in top_matches]
    
    # Save the career and matched careers to the results list
    results.append([small_career, ", ".join(matched_careers)])

# Convert the results to a DataFrame
output_df = pd.DataFrame(results, columns=["Career (Small List)", "Matched Careers (Large List)"])

# Save the result to a CSV file
output_df.to_csv("matched_careers.csv", index=False)

print("Matching complete! Results saved to matched_careers.csv")
