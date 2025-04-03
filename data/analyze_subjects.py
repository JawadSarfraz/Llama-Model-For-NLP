import json
from collections import Counter

def analyze_subjects():
    # Load the dataset
    with open('data/sample_data.json', 'r') as f:
        data = json.load(f)
    
    # Initialize counters
    total_papers = len(data)
    subjects_per_paper = []
    all_subjects = []
    
    # Analyze each paper
    for paper in data:
        if 'subject' in paper:
            paper_subjects = paper['subject']
            subjects_per_paper.append(len(paper_subjects))
            all_subjects.extend(paper_subjects)
    
    # Count subject frequencies
    subject_counts = Counter(all_subjects)
    
    # Print analysis
    print(f"\nDataset Analysis:")
    print(f"Total papers: {total_papers}")
    print(f"\nSubjects per paper:")
    print(f"Average: {sum(subjects_per_paper)/len(subjects_per_paper):.2f}")
    print(f"Min: {min(subjects_per_paper)}")
    print(f"Max: {max(subjects_per_paper)}")
    
    print(f"\nUnique subjects: {len(subject_counts)}")
    print("\nTop 10 most common subjects:")
    for subject, count in subject_counts.most_common(10):
        print(f"{subject}: {count} papers ({count/total_papers*100:.1f}%)")
    
    # Distribution of number of subjects per paper
    subject_count_dist = Counter(subjects_per_paper)
    print("\nDistribution of subjects per paper:")
    for num_subjects, count in sorted(subject_count_dist.items()):
        print(f"{num_subjects} subjects: {count} papers ({count/total_papers*100:.1f}%)")

if __name__ == "__main__":
    analyze_subjects() 