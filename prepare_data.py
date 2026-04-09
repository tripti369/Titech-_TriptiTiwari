import pandas as pd

# This script simulates the "Explanation" part. 
# In a real project, you'd use a large AI to generate these.
def create_training_pairs():
    df = pd.read_csv("data/raw_clauses.csv")
    
    # Placeholder: We are adding a dummy explanation.
    # Replace this logic with actual AI-generated summaries.
    df['explanation'] = "This clause outlines the rules and responsibilities for both the landlord and the tenant regarding this specific section of the lease."
    
    df.to_csv("data/paired_data.csv", index=False)
    print("Training data prepared: data/paired_data.csv")

if __name__ == "__main__":
    create_training_pairs()