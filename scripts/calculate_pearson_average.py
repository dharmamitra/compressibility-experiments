import pandas as pd
import numpy as np
from scipy import stats

# Function to read TSV file
def read_tsv(file_path):
    return pd.read_csv(file_path, sep='\t')

# Read compression ratios
compression_file = 'compression_ratios.csv'  # Update this path
compression_df = pd.read_csv(compression_file)

# Read LLM scores
scores_file = 'updated_merged_output.tsv'  # Update this path
scores_df = read_tsv(scores_file)

# Calculate average scores across LLMs for each document and score type
avg_scores = scores_df.groupby('Document').agg({
    'CHRF Score': 'mean',
    'BLEU Score': 'mean',
    'BERT Score': 'mean',
    'BLEURT Score': 'mean'
}).reset_index()

# Merge with compression ratios
merged_df = pd.merge(avg_scores, compression_df, left_on='Document', right_on='File')

# Calculate correlations
score_types = ['CHRF Score', 'BLEU Score', 'BERT Score', 'BLEURT Score']

results = []

for score_type in score_types:
    correlation, p_value = stats.pearsonr(merged_df[score_type], merged_df['Ratio'])
    results.append({
        'Score Type': score_type,
        'Correlation': correlation,
        'P-value': p_value
    })

results_df = pd.DataFrame(results)

# Sort results by absolute correlation value
results_df['Abs Correlation'] = results_df['Correlation'].abs()
results_df = results_df.sort_values('Abs Correlation', ascending=False)

# Display results
print("Correlation of average scores with compressibility:")
print(results_df[['Score Type', 'Correlation', 'P-value']])

# Optionally, save results to a file
results_df.to_csv('average_correlation_results.csv', index=False)
print("Results saved to 'average_correlation_results.csv'")

# Display average scores for each document
print("\nAverage scores for each document:")
print(avg_scores)

# Optionally, save average scores to a file
avg_scores.to_csv('average_scores.csv', index=False)
print("Average scores saved to 'average_scores.csv'")