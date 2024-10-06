from transformers import RobertaForQuestionAnswering, RobertaTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load the model and tokenizer
model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2")

# Load the data
dataset = load_dataset("json", data_files={"train": "retrained_1.json", "validation": "validation.json"})

def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples['question'], 
        examples['context'],  # Use context instead of answer
        truncation=True, 
        padding="max_length", 
        max_length=512  # Adjust max_length according to your needs
    )
    
    start_positions = []
    end_positions = []

    # Iterate through questions
    for i in range(len(examples['question'])):
        start_positions.append(examples['answers'][i]['start'])  # Assuming there's only one answer
        end_positions.append(examples['answers'][i]['end'])

    tokenized_inputs['start_positions'] = start_positions
    tokenized_inputs['end_positions'] = end_positions

    return tokenized_inputs

# Tokenize the datasets
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Verify lengths
print("Lengths of the examples:")
for item in tokenized_dataset['train']:
    print(len(item['input_ids']))

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_Roberta")