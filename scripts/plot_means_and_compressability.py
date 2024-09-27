import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_visualization(score_type):
    # Read the data
    scores_df = pd.read_csv('updated_merged_output.tsv', sep='\t')
    compressibility_df = pd.read_csv('compression_ratios.csv')

    # Calculate mean scores for the specific score type across LLMs for each document
    mean_scores = scores_df.groupby('Document')[score_type].mean().reset_index()

    # Merge the dataframes
    merged_df = pd.merge(mean_scores, compressibility_df, left_on='Document', right_on='File')

    # Sort the dataframe by compressibility ratio
    merged_df = merged_df.sort_values('Ratio', ascending=False)

    # Create a list of documents in order of compressibility
    documents = merged_df['Document']

    # Set up the plot style
    plt.style.use('ggplot')

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Compressibility Ratios and Average {score_type} Across Documents', fontsize=16)

    # Plot compressibility ratios
    ax1.plot(documents, merged_df['Ratio'], marker='o', linestyle='-', color='black', label='Compressibility Ratio')
    ax1.set_ylabel('Compressibility Ratio')
    ax1.set_title('Compressibility Ratios')
    ax1.legend()

    # Plot average scores
    ax2.plot(documents, merged_df[score_type], marker='o', linestyle='-', label=f'Average {score_type}', color='blue')
    ax2.set_ylabel(f'Average {score_type}')
    ax2.set_title(f'Average {score_type} Across Documents')
    ax2.legend()

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'compressibility_{score_type.lower().replace(" ", "_")}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved as 'compressibility_{score_type.lower().replace(' ', '_')}_comparison.png'")

# Create visualizations for each score type
score_types = ['CHRF Score', 'BLEU Score', 'BERT Score', 'BLEURT Score']
for score_type in score_types:
    create_visualization(score_type)