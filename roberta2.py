import json
from datasets import load_dataset

# Fix the schema of the dataset to match the required structure
def load_corrected_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        corrected_data = []
        for item in data["questions"]:
            corrected_data.append({
                
                    "question": item["question"],
                    "answer": item["answer"]
                }
            )
        return corrected_data

# Load the retrained and validation datasets
train_data = load_corrected_json("retrained_1.json")
validation_data = load_corrected_json("validation.json")

# Save the corrected data to temporary JSON files to use with load_dataset
with open("corrected_train.json", 'w') as train_file:
    json.dump({"questions": train_data}, train_file)
with open("corrected_validation.json", 'w') as validation_file:
    json.dump({"questions": validation_data}, validation_file)

# Now use the load_dataset function
dataset = load_dataset("json", data_files={"train": "corrected_train.json", "validation": "corrected_validation.json"})
