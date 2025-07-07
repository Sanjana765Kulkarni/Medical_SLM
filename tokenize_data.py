import os
from transformers import AutoTokenizer

def tokenize_text_data(data_folder, output_file, model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    all_text = []
    for filename in os.listdir(data_folder):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(data_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_text.append(f.read())
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    combined_text = "\n".join(all_text)
    
    # Tokenize the combined text
    # We use return_tensors="pt" to get PyTorch tensors, which is common for training
    # We also set truncation=True to handle very long texts, though for training, 
    # we'll typically break them into smaller chunks later.
    tokenized_inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    
    # Save the input IDs (the tokenized numerical representation)
    import torch
    torch.save(tokenized_inputs['input_ids'], output_file)
    print(f"Tokenized data saved to {output_file}")

if __name__ == "__main__":
    data_folder = r'E:\the slm project\data'
    output_file = r'E:\the slm project\tokenized_data.pt'
    
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found at {data_folder}")
    else:
        tokenize_text_data(data_folder, output_file)