

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import os

def train_slm(tokenized_data_path, output_dir="./slm_model", model_name="gpt2", block_size=128):
    # Load tokenized data
    input_ids = torch.load(tokenized_data_path)
    
    # Flatten the input_ids to a 1D tensor
    token_ids_tensor = input_ids.flatten()

    # Chunk the token_ids_tensor into blocks
    # This creates a list of dictionaries, where each dictionary is an example for the dataset
    # We discard the last chunk if it's smaller than block_size
    examples = []
    for i in range(0, len(token_ids_tensor) - block_size + 1, block_size):
        examples.append({"input_ids": token_ids_tensor[i : i + block_size]})

    # Create a Dataset from the list of examples
    dataset = Dataset.from_list(examples)

    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad_token_id for GPT-2 if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Data collator for language modeling
    # It will take care of chunking the long sequence into blocks and creating labels
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # False for causal language modeling (like GPT-2)
    )

    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Ensure model's pad_token_id is set correctly if tokenizer's was changed
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,  # You can adjust this
        per_device_train_batch_size=4, # Adjust based on your GPU memory
        save_steps=10_000, # Save checkpoint every 10,000 steps
        save_total_limit=2, # Only keep the last 2 checkpoints
        logging_dir='./logs',
        logging_steps=500,
        report_to="none", # Disable reporting to services like Weights & Biases
        # Add these for better GPU utilization and to avoid potential issues
        fp16=torch.cuda.is_available(), # Use mixed precision training if GPU is available
        gradient_accumulation_steps=8, # Accumulate gradients over multiple steps to simulate larger batch size
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer, # Keep tokenizer here for internal use by Trainer
        data_collator=data_collator,
    )

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training complete. Saving model...")

    # Save the fine-tuned model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    tokenized_data_path = r'E:\the slm project\tokenized_data.pt'
    output_model_dir = r'E:\the slm project\slm_model'
    
    # Ensure output directory exists
    os.makedirs(output_model_dir, exist_ok=True)

    if not os.path.exists(tokenized_data_path):
        print(f"Error: Tokenized data not found at {tokenized_data_path}")
    else:
        train_slm(tokenized_data_path, output_model_dir)
