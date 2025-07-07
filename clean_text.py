
import os
import re

def clean_text(text):
    # Remove page numbers (e.g., - 123 - or 123)
    text = re.sub(r'\s*-\s*\d+\s*-\s*|\s+\d+\s*\n', '\n', text)
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n', text)
    # Remove hyphenation at line breaks
    text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text, flags=re.IGNORECASE)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    data_folder = r'E:\the slm project\data'
    
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found at {data_folder}")
        return

    for filename in os.listdir(data_folder):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(data_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                cleaned_content = clean_text(content)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                print(f"Cleaned and overwrote {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
