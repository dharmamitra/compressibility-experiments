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

scores_df = scores_df[scores_df['Document'] != 'all_files.tsv']

# Merge dataframes
merged_df = pd.merge(scores_df, compression_df, left_on='Document', right_on='File')
merged_df[['Document', 'Ratio']].drop_duplicates()
print(merged_df)

# Calculate correlations
llms = merged_df['LLM'].unique()
score_types = ['CHRF Score', 'BLEU Score', 'BERT Score', 'BLEURT Score']

results = []

for llm in llms:
    llm_df = merged_df[merged_df['LLM'] == llm]
    for score_type in score_types:
        correlation, p_value = stats.pearsonr(llm_df[score_type], llm_df['Ratio'])
        results.append({
            'LLM': llm,
            'Score Type': score_type,
            'Correlation': correlation,
            'P-value': p_value
        })


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(['Score Type', 'LLM'])

output_file = 'correlation_results_new.csv'

results_df.to_csv(output_file, index=False)
print("Results saved to "+ output_file)