import PyPDF2
import os

def extract_text_from_pdf(pdf_path, output_folder):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        
        # Create output filename
        pdf_filename = os.path.basename(pdf_path)
        txt_filename = os.path.splitext(pdf_filename)[0] + '.txt'
        output_path = os.path.join(output_folder, txt_filename)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(text)
        print(f"Extracted text from {pdf_filename} to {txt_filename}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

def main():
    data_folder = r'E:\the slm project\data' # Use raw string for path
    
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found at {data_folder}")
        return

    for filename in os.listdir(data_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(data_folder, filename)
            extract_text_from_pdf(pdf_path, data_folder)

if __name__ == "__main__":
    main()
