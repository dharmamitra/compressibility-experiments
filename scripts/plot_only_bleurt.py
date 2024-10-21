import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

# Read the data
scores_df = pd.read_csv('output_filtered.tsv', sep='\t')
compressibility_df = pd.read_csv('marcus_compression_ratios.csv')

# extract document identifier in form of T02n0099
scores_df['Document'] = scores_df['Document'].str.extract(r'(T\d+n\d+)')
compressibility_df['File'] = compressibility_df['File'].str.extract(r'/(T\d+n\d+)')


# Merge the dataframes
merged_df = pd.merge(scores_df, compressibility_df, left_on='Document', right_on='File')

# Function to simplify document names
def simplify_document_name(name):
    parts = name.split('_')
    if len(parts) > 1:
        return ' '.join(parts[1:]).rsplit('.', 1)[0]  # Remove file extension if present
    return name

# Apply the simplification to the Document column
merged_df['Document'] = merged_df['Document'].apply(simplify_document_name)

# Increase the default font size
plt.rcParams.update({'font.size': 14})

# Create the figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(20, 15), sharex=True)
#fig.suptitle('Compressibility Ratios and BLEURT Scores Across Documents', fontsize=24, y=0.5)

# Plot compressibility ratios
compressibility_data = merged_df.groupby('Document')['Ratio'].first().sort_values(ascending=False)
axs[0].plot(range(len(compressibility_data)), compressibility_data.values, 
            marker='o', linestyle='-', color='black', label='Compressibility Ratio')
axs[0].set_ylabel('Compressibility Ratio', fontsize=18)
axs[0].set_title('Compressibility Ratios', fontsize=20)
axs[0].legend(fontsize=16)
axs[0].set_xticks(range(len(compressibility_data)))
axs[0].set_xticklabels(compressibility_data.index, rotation=45, ha='right', fontsize=14)

# Plot BLEURT scores
avg_scores = merged_df.groupby(['Document', 'LLM'])['BLEURT Score'].mean().unstack()

# Sort documents by highest average score
document_order = avg_scores.mean(axis=1).sort_values(ascending=False).index

# Sort LLMs by highest average score
llm_order = avg_scores.mean().sort_values(ascending=False).index

# Assign colors to LLMs
colors = plt.cm.Set2(np.linspace(0, 1, len(llm_order)))
color_dict = dict(zip(llm_order, colors))

for llm in llm_order:
    llm_data = merged_df[merged_df['LLM'] == llm].set_index('Document')
    sorted_data = llm_data.loc[document_order, 'BLEURT Score']
    axs[1].plot(range(len(document_order)), sorted_data, 
                marker='o', linestyle='-', label=llm, color=color_dict[llm])

axs[1].set_ylabel('BLEURT Score', fontsize=18)
axs[1].set_title('BLEURT Scores Across Documents', fontsize=20)
axs[1].set_xticks(range(len(document_order)))
axs[1].set_xticklabels(document_order, rotation=45, ha='right', fontsize=14)

# Sort legend by highest average score
handles, labels = axs[1].get_legend_handles_labels()
sorted_pairs = sorted(zip(handles, labels), 
                      key=lambda x: avg_scores[x[1]].mean(), 
                      reverse=True)
sorted_handles, sorted_labels = zip(*sorted_pairs)
axs[1].legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=16)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.2, left=0.1)  # Increased left margin

output_file = 'bleurt_ratio_marcus.png'
# Save the figure
plt.savefig(output_file, dpi=300, )
plt.close()

print("Visualization saved as " + output_file)

