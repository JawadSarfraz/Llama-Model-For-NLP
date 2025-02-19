import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 2: Load JSON data from the file
def load_data():
    with open('../data/sample_data.json', 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

# Step 3: Preprocess and split the data
def prepare_data(df):
    # Convert list of subjects to a single string (assuming each list has only one item)
    df['subject'] = df['subject'].apply(lambda x: x[0] if x else None)
    
    # Assuming 'abstract' fields are lists of strings
    df['text'] = df['abstract'].apply(lambda x: ' '.join(x).lower())
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['subject'])
    return train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Step 4: Main execution function
def main():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    # Add your model training and evaluation code here

if __name__ == '__main__':
    main()