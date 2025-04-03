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
    unique_subjects = set()  # To track truly unique subjects
    
    # Analyze each paper
    for paper in data:
        if 'subject' in paper:
            paper_subjects = paper['subject']
            subjects_per_paper.append(len(paper_subjects))
            # Add to all_subjects for frequency counting
            all_subjects.extend(paper_subjects)
            # Add to unique_subjects set
            unique_subjects.update(paper_subjects)
    
    # Count subject frequencies
    subject_counts = Counter(all_subjects)
    
    # Print analysis
    print(f"\nDataset Analysis:")
    print(f"Total papers: {total_papers}")
    print(f"\nSubjects per paper:")
    print(f"Average: {sum(subjects_per_paper)/len(subjects_per_paper):.2f}")
    print(f"Min: {min(subjects_per_paper)}")
    print(f"Max: {max(subjects_per_paper)}")
    
    print(f"\nTruly unique subjects (case-sensitive): {len(unique_subjects)}")
    print(f"Case-insensitive unique subjects: {len({s.lower() for s in unique_subjects})}")
    
    print("\nTop 20 most common subjects:")
    print("Subject (frequency) [percentage of papers]")
    print("-" * 50)
    for subject, count in subject_counts.most_common(20):
        print(f"{subject} ({count}) [{count/total_papers*100:.1f}%]")
    
    # Distribution of number of subjects per paper
    subject_count_dist = Counter(subjects_per_paper)
    print("\nDistribution of subjects per paper:")
    for num_subjects, count in sorted(subject_count_dist.items()):
        print(f"{num_subjects} subjects: {count} papers ({count/total_papers*100:.1f}%)")
    
    # Print some example papers with their subjects
    print("\nExample papers with subjects:")
    print("-" * 50)
    for i, paper in enumerate(data[:3]):  # Show first 3 papers
        if 'subject' in paper:
            print(f"\nPaper {i+1}:")
            print(f"Number of subjects: {len(paper['subject'])}")
            print("Subjects:", ", ".join(paper['subject']))

if __name__ == "__main__":
    analyze_subjects() 