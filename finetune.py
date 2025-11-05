from sklearn.model_selection import train_test_split
import pandas as pd
import re
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split

# Load the CSV dataset
dataset = pd.read_csv('counselchat_data.csv')

# Preprocess the dataset
def preprocess_example(example):
    example['user'] = re.sub(r'[^\w\s\'",!?]', '', example['user'])
    example['therapist'] = re.sub(r'[^\w\s\'",!?]', '', example['therapist'])
    return example

dataset = dataset.map(preprocess_example)

# Split dataset into train and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

# Load the tokenizer and model
tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# Define the data collator
data_collator = DataCollatorForSeq2Seq.from_pretrained(tokenizer, model=model)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('fine-tuned-model')

# Load the fine-tuned model
model = BlenderbotForConditionalGeneration.from_pretrained('fine-tuned-model')

# Generate responses for some example inputs
inputs = ["user: Hello", "user: How are you?"]
for input_text in inputs:
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    output_ids = model.generate(input_ids)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Bot:", response)
