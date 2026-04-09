import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def train():
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(r=8, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("csv", data_files="data/paired_data.csv", split="train")

    formatted_texts = []
    for example in dataset:
        formatted_texts.append(f"Clause: {example['legal_clause']}\nSummary: {example['explanation']}")

    # Create new dataset with text
    from datasets import Dataset
    text_dataset = Dataset.from_dict({"text": formatted_texts})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = text_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./models/gpt2-ft",
            num_train_epochs=1,
            logging_steps=1,
            per_device_train_batch_size=1,
            max_steps=60,
            push_to_hub=False,
        ),
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    model.save_pretrained("./models/final_model")
    print("Training Complete! Model saved to ./models/final_model")

if __name__ == "__main__":
    train()