import pandas as pd
import numpy as np
from scipy import stats

# Read compression ratios
compression_df = pd.read_csv('compression_ratios.csv')

# Read LLM scores
scores_df = pd.read_csv('updated_merged_output.tsv', sep='\t')

# Remove 'all_files.tsv' entries as they don't have a corresponding compression ratio
scores_df = scores_df[scores_df['Document'] != 'all_files.tsv']

# Merge dataframes
merged_df = pd.merge(scores_df, compression_df, left_on='Document', right_on='File', how='inner')

print("Merged DataFrame sample:")
print(merged_df.head())

print("\nUnique Documents and their Ratios:")
print(merged_df[['Document', 'Ratio']].drop_duplicates())

llms = merged_df['LLM'].unique()
score_types = ['CHRF Score', 'BLEU Score', 'BERT Score', 'BLEURT Score']

# Calculate correlations for each LLM
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
print("\nCorrelation Results:")
print(results_df)

# Calculate overall correlations (across all LLMs)
print("\nOverall Correlations:")
for score_type in score_types:
    correlation, p_value = stats.pearsonr(merged_df[score_type], merged_df['Ratio'])
    print(f"{score_type}: Correlation = {correlation:.4f}, P-value = {p_value:.4f}")

# Analyze score distributions
print("\nScore Distributions:")
for score_type in score_types:
    print(f"\n{score_type}:")
    for llm in llms:
        llm_scores = merged_df[merged_df['LLM'] == llm][score_type]
        print(f"  {llm}: Mean = {llm_scores.mean():.4f}, Std = {llm_scores.std():.4f}")

# Visualize the relationship between scores and compression ratios
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
for llm in llms:
    llm_df = merged_df[merged_df['LLM'] == llm]
    plt.scatter(llm_df['Ratio'], llm_df['BERT Score'], label=llm, alpha=0.7)

plt.xlabel('Compression Ratio')
plt.ylabel('BERT Score')
plt.title('BERT Score vs Compression Ratio for different LLMs')
plt.legend()
plt.savefig('bert_vs_ratio.png')
plt.close()

print("\nA scatter plot of BERT Score vs Compression Ratio has been saved as 'bert_vs_ratio.png'")