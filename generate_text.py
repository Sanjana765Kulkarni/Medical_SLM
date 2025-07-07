import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(model_path, prompt, max_length=100, num_return_sequences=1):
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2, # Avoid repeating n-grams
        do_sample=True, # Use sampling for more diverse output
        top_k=50, # Sample from top 50 most likely tokens
        top_p=0.95, # Sample from top tokens whose cumulative probability exceeds 0.95
        temperature=0.7, # Control randomness
        pad_token_id=tokenizer.eos_token_id # Use EOS token as pad token
    )

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    model_path = r'E:\the slm project\slm_model'
    
    print("Enter your prompt (type 'exit' to quit):\n")
    while True:
        user_prompt = input("> ")
        if user_prompt.lower() == 'exit':
            break
        
        if user_prompt:
            generate_text(model_path, user_prompt)
        else:
            print("Please enter a non-empty prompt.")
