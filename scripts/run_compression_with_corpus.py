import os
import gzip
import glob
import csv

def get_gzip_size(content):
    return len(gzip.compress(content.encode('utf-8')))

def process_file(ref_content, ref_gzip_size, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Keep only the Chinese text (first column)
    chinese_text = '\n'.join(line.split('\t')[0] for line in lines)
    
    # Calculate original file size
    orig_file_size = len(chinese_text.encode('utf-8'))
    
    # Compress reference corpus + current text
    ref_plus_current_text_gzip_size = get_gzip_size(ref_content + chinese_text)
    
    # Calculate difference in size
    diff_size = abs(ref_gzip_size - ref_plus_current_text_gzip_size)
    
    # Calculate ratio
    ratio = orig_file_size / diff_size if diff_size != 0 else float('inf')
    
    return orig_file_size, diff_size, ratio

def main():
    # Read reference corpus
    with open('data/chn_refence_corpus.txt', 'r', encoding='utf-8') as f:
        ref_content = f.read()
    
    # Get gzip size of reference corpus
    ref_gzip_size = get_gzip_size(ref_content)
    
    # Process all TSV files in the zh directory
    results = []
    for file_path in glob.glob('zh/*.tsv'):
        orig_size, diff_size, ratio = process_file(ref_content, ref_gzip_size, file_path)
        results.append((file_path, orig_size, diff_size, ratio))
    
    # Sort results by ratio in descending order
    results.sort(key=lambda x: x[3], reverse=True)
    
    # Print results to console
    print("File\tOriginal Size\tDiff Size\tRatio")
    for file_path, orig_size, diff_size, ratio in results:
        print(f"{file_path}\t{orig_size}\t{diff_size}\t{ratio:.2f}")
    
    # Save results to a CSV file
    output_file = 'compression_ratios.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['File', 'Original Size', 'Diff Size', 'Ratio'])
        for file_path, orig_size, diff_size, ratio in results:
            csvwriter.writerow([file_path, orig_size, diff_size, f"{ratio:.2f}"])
    
    print(f"\nResults have been saved to {output_file}")

if __name__ == "__main__":
    main()