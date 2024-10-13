import os
import gzip
import glob
import csv

def get_gzip_size(content):
    return len(gzip.compress(content.encode('utf-8')))

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Keep only the Chinese text (first column)
    chinese_text = '\n'.join(line.split('\t')[0] for line in lines)
    
    # Calculate original file size
    orig_file_size = len(chinese_text.encode('utf-8'))
    
    # Calculate gzipped size
    gzip_size = get_gzip_size(chinese_text)
    
    # Calculate ratio
    ratio = orig_file_size / gzip_size if gzip_size != 0 else float('inf')
    
    return orig_file_size, gzip_size, ratio

def main():
    # Process all TSV files in the zh directory
    results = []
    for file_path in glob.glob('zh/*.tsv'):
        orig_size, gzip_size, ratio = process_file(file_path)
        results.append((file_path, orig_size, gzip_size, ratio))
    
    # Sort results by ratio in descending order
    results.sort(key=lambda x: x[3], reverse=True)
    
    # Print results to console
    print("File\tOriginal Size\tGzip Size\tRatio")
    for file_path, orig_size, gzip_size, ratio in results:
        print(f"{file_path}\t{orig_size}\t{gzip_size}\t{ratio:.2f}")
    
    # Save results to a CSV file
    output_file = 'compression_ratios_self.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['File', 'Original Size', 'Gzip Size', 'Ratio'])
        for file_path, orig_size, gzip_size, ratio in results:
            csvwriter.writerow([file_path, orig_size, gzip_size, f"{ratio:.2f}"])
    
    print(f"\nResults have been saved to {output_file}")

if __name__ == "__main__":
    main()