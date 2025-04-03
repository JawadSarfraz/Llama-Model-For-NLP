import json
from collections import Counter
import matplotlib.pyplot as plt

def analyze_frequent_subjects(min_papers=20):
    """
    Analyze subjects that appear in at least min_papers number of papers.
    """
    # Load the dataset
    with open('data/sample_data.json', 'r') as f:
        data = json.load(f)
    
    # Count subject frequencies
    all_subjects = []
    for paper in data:
        if 'subject' in paper:
            all_subjects.extend(paper['subject'])
    
    subject_counts = Counter(all_subjects)
    total_papers = len(data)
    
    # Filter subjects that appear in more than min_papers
    frequent_subjects = {subject: count 
                        for subject, count in subject_counts.items() 
                        if count >= min_papers}
    
    # Print analysis
    print(f"\nSubject Frequency Analysis (minimum {min_papers} papers):")
    print(f"Total unique subjects: {len(subject_counts)}")
    print(f"Subjects appearing in {min_papers}+ papers: {len(frequent_subjects)}")
    
    print(f"\nTop frequent subjects (appearing in {min_papers}+ papers):")
    print(f"{'Subject':<50} | {'Count':>6} | {'% of Papers':>10}")
    print("-" * 70)
    
    for subject, count in sorted(frequent_subjects.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_papers) * 100
        print(f"{subject:<50} | {count:>6} | {percentage:>9.1f}%")
    
    # Calculate coverage
    papers_with_frequent = 0
    papers_with_only_frequent = 0
    
    for paper in data:
        if 'subject' in paper:
            paper_subjects = set(paper['subject'])
            frequent_paper_subjects = paper_subjects.intersection(frequent_subjects.keys())
            
            if frequent_paper_subjects:
                papers_with_frequent += 1
                if frequent_paper_subjects == paper_subjects:
                    papers_with_only_frequent += 1
    
    print(f"\nCoverage Analysis:")
    print(f"Papers with at least one frequent subject: {papers_with_frequent} ({papers_with_frequent/total_papers*100:.1f}%)")
    print(f"Papers with only frequent subjects: {papers_with_only_frequent} ({papers_with_only_frequent/total_papers*100:.1f}%)")
    
    # Distribution of subject frequencies
    counts = list(subject_counts.values())
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=50, log=True)
    plt.title('Distribution of Subject Frequencies')
    plt.xlabel('Number of Papers')
    plt.ylabel('Number of Subjects (log scale)')
    plt.axvline(x=min_papers, color='r', linestyle='--', label=f'Threshold ({min_papers} papers)')
    plt.legend()
    plt.savefig('data/subject_distribution.png')
    plt.close()
    
    return frequent_subjects

if __name__ == "__main__":
    # Try different thresholds
    for threshold in [20, 30, 50]:
        print(f"\n{'='*80}")
        print(f"Analysis for threshold: {threshold} papers")
        print(f"{'='*80}")
        frequent_subjects = analyze_frequent_subjects(threshold) 