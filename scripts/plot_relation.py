import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the data
scores_df = pd.read_csv('updated_merged_output.tsv', sep='\t')
compressibility_df = pd.read_csv('compression_ratios_self.csv')

# Merge the dataframes
merged_df = pd.merge(scores_df, compressibility_df, left_on='Document', right_on='File')

# Set up the plot style
#plt.style.use('ggplot')

# Create the figure and subplots
fig, axs = plt.subplots(5, 1, figsize=(14, 6), sharex=True)
fig.suptitle('Compressibility Ratios and Scores Across Documents', fontsize=16, y=0.95)

# Plot compressibility ratios
compressibility_data = merged_df.groupby('Document')['Ratio'].first().sort_values(ascending=False)
axs[0].plot(range(len(compressibility_data)), compressibility_data.values, 
            marker='o', linestyle='-', color='black', label='Compressibility Ratio')
axs[0].set_ylabel('Compressibility Ratio')
axs[0].set_title('Compressibility Ratios')
axs[0].legend()
axs[0].set_xticks(range(len(compressibility_data)))
axs[0].set_xticklabels(compressibility_data.index, rotation=45, ha='right')

# Plot scores
score_types = ['BLEURT Score', 'BERT Score', 'CHRF Score', 'BLEU Score']

for i, score_type in enumerate(score_types, start=1):
    # Calculate average scores for each document and LLM
    avg_scores = merged_df.groupby(['Document', 'LLM'])[score_type].mean().unstack()
    
    # Sort documents by highest average score
    document_order = avg_scores.mean(axis=1).sort_values(ascending=False).index
    
    # Sort LLMs by highest average score
    llm_order = avg_scores.mean().sort_values(ascending=False).index
    
    # Assign colors to LLMs
    colors = plt.cm.Set2(np.linspace(0, 1, len(llm_order)))
    color_dict = dict(zip(llm_order, colors))
    
    for llm in llm_order:
        llm_data = merged_df[merged_df['LLM'] == llm].set_index('Document')
        sorted_data = llm_data.loc[document_order, score_type]
        axs[i].plot(range(len(document_order)), sorted_data, 
                    marker='o', linestyle='-', label=llm, color=color_dict[llm])
    
    axs[i].set_ylabel(score_type)
    axs[i].set_title(f'{score_type} Across Documents')
    axs[i].set_xticks(range(len(document_order)))
    axs[i].set_xticklabels(document_order, rotation=45, ha='right')
    
    # Sort legend by highest average score
    handles, labels = axs[i].get_legend_handles_labels()
    sorted_pairs = sorted(zip(handles, labels), 
                          key=lambda x: avg_scores[x[1]].mean(), 
                          reverse=True)
    sorted_handles, sorted_labels = zip(*sorted_pairs)
    axs[i].legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4)


output_file = 'new_comparison_self.png'
# Save the figure
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved as " + output_file)