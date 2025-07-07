# Project Report: The SLM Project

**1. Project Goal:**
The primary objective is to build a Small Language Model (SLM) capable of understanding and generating text related to the first-year medical school curriculum.

**2. Project Setup:**
*   A dedicated project directory, `E:\the slm project`, was created.
*   A `README.md` file was initialized to outline the project's purpose.

**3. Data Acquisition & Preparation:**
*   **Curriculum Identification:** Core subjects for first-year medical school (Anatomy, Physiology, Biochemistry, etc.) were identified through web search.
*   **Textbook Sourcing:** Open-source textbooks from OpenStax and other open educational resources were identified as primary data sources. The user manually downloaded these PDF files into the `E:\the slm project\data` directory.
*   **Text Extraction:** The `PyPDF2` library was installed, and a Python script (`extract_text.py`) was created and executed to convert all downloaded PDF files into raw text (`.txt`) files.
*   **Text Cleaning:** A `clean_text.py` script was developed and executed to remove common PDF artifacts (page numbers, hyphenation, extra spaces) from the extracted text, preparing it for model training.
*   **Tokenization:** The cleaned text was tokenized using a GPT-2 tokenizer via `tokenize_data.py`, converting it into numerical representations (`tokenized_data.pt`) suitable for a language model.

**4. Development Environment Setup:**
*   A Python virtual environment named `slm_env` was created within the project directory to manage dependencies.
*   Key machine learning libraries (`torch`, `transformers`, `datasets`, `accelerate`) were installed into this virtual environment.
*   **GPU Configuration:** Initial issues with PyTorch not detecting the GPU were resolved by reinstalling the correct CUDA-enabled version of `torch` (compatible with CUDA 12.2).

**5. Model Training:**
*   A `train_slm.py` script was developed to fine-tune a pre-trained GPT-2 model on the tokenized medical data.
*   **Troubleshooting:** An `IndexError` during training was resolved by implementing `DataCollatorForLanguageModeling` and explicitly chunking the tokenized data into fixed-size blocks, ensuring correct input shape for the model.
*   **Training Outcome:** The model completed 3 training epochs. The observed `train_loss` was `3.77`, which is relatively high, indicating that the model has only begun to learn the specialized medical domain. The fine-tuned model was saved to `E:\the slm project\slm_model`.

**6. Model Inference (Text Generation):**
*   A `generate_text.py` script was created to allow interactive text generation from the trained SLM.
*   **Current Capabilities:** Initial testing showed the model can generate text related to the prompt but quickly deviates into irrelevant or nonsensical terms, reflecting the high training loss and limited exposure to the specialized medical vocabulary.

**Next Steps:**
To significantly improve the SLM's performance and reduce the training loss, the primary focus should be on:
*   **Increasing Data Volume:** Acquiring and processing more high-quality medical text data.
*   **Extended Training:** Training the model for more epochs.
*   **Hyperparameter Tuning:** Experimenting with training parameters.
