# The SLM Project: Medical Domain Small Language Model

## Overview
This project aims to develop a Small Language Model (SLM) specifically fine-tuned on first-year medical school curriculum data. The goal is to create a compact language model capable of understanding and generating text related to foundational medical sciences.

## Features
-   **Data Preparation Pipeline:** Scripts for extracting text from PDF textbooks, cleaning the raw text, and tokenizing it for model input.
-   **SLM Training:** Fine-tuning of a pre-trained GPT-2 model on specialized medical text data using PyTorch and Hugging Face Transformers.
-   **GPU Acceleration:** Configured to leverage NVIDIA GPUs (CUDA) for efficient training.
-   **Text Generation (Inference):** An interactive script to generate medical-related text based on user prompts.

## Getting Started

### Prerequisites
-   Python 3.8+
-   NVIDIA GPU with CUDA 11.x or 12.x installed (for GPU acceleration)
-   `pip` (Python package installer)

### Setup
1.  **Clone the repository (or navigate to your project directory):**
    ```bash
    cd "E:\the slm project"
    ```

2.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv slm_env
    ```

3.  **Activate the Virtual Environment:**
    *   **Windows (PowerShell):**
        ```powershell
        . "E:\the slm project\slm_env\Scripts\Activate.ps1"
        ```
    *   **Windows (Command Prompt):**
        ```cmd
        E:\the slm project\slm_env\Scripts\activate.bat
        ```
    *   **macOS/Linux:**
        ```bash
        source E:/the slm project/slm_env/bin/activate
        ```
    *(Ensure your prompt changes to `(slm_env)`)*

4.  **Install Dependencies:**
    If you encounter issues with direct activation, you can install dependencies using the virtual environment's Python executable directly:
    ```powershell
    & "E:\the slm project\slm_env\Scripts\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    & "E:\the slm project\slm_env\Scripts\python.exe" -m pip install transformers datasets accelerate PyPDF2
    ```
    *(Note: `cu121` is compatible with CUDA 12.2. Adjust `--index-url` if your CUDA version differs.)*

## Data Preparation

1.  **Download Medical Textbooks:**
    Acquire PDF versions of first-year medical textbooks. Recommended open-source sources include:
    *   **Anatomy & Physiology:** [OpenStax Anatomy and Physiology](https://openstax.org/details/books/anatomy-and-physiology)
    *   **Biochemistry:** [The Basics of General, Organic, and Biological Chemistry](https://open.umn.edu/opentextbooks/textbooks/the-basics-of-general-organ
ic-and-biological-chemistry)
    *   **Genetics:** [OpenStax Biology 2e](https://openstax.org/details/books/biology-2e) (relevant chapters)
    *   **Pharmacology:** [Nursing Pharmacology](https://open.umn.edu/opentextbooks/textbooks/nursing-pharmacology) (foundational)
    *   **Neuroscience:** [OpenStax Introduction to Behavioral Neuroscience](https://openstax.org/details/books/introduction-behavioral-neuroscience)
    *   Place all downloaded PDF files into the `E:\the slm project\data` directory.

2.  **Extract Text from PDFs:**
    ```bash
    & "E:\the slm project\slm_env\Scripts\python.exe" "E:\the slm project\extract_text.py"
    ```
    This will convert PDFs to `.txt` files in the `data` directory.

3.  **Clean Extracted Text:**
    ```bash
    & "E:\the slm project\slm_env\Scripts\python.exe" "E:\the slm project\clean_text.py"
    ```
    This will clean the `.txt` files in-place.

4.  **Tokenize Data:**
    ```bash
    & "E:\the slm project\slm_env\Scripts\python.exe" "E:\the slm project\tokenize_data.py"
    ```
    This will create `tokenized_data.pt` in the project root.

## Training the SLM

To fine-tune the GPT-2 model on your prepared medical data:

```bash
& "E:\the slm project\slm_env\Scripts\python.exe" "E:\the slm project\train_slm.py"
```
*(Training can take a significant amount of time depending on your hardware and data volume.)*
The trained model will be saved in the `E:\the slm project\slm_model` directory.

## Text Generation (Inference)

Once the model is trained, you can generate text interactively:

```bash
& "E:\the slm project\slm_env\Scripts\python.exe" "E:\the slm project\generate_text.py"
```
Follow the prompts to enter your text and see the model's output. Type `exit` to quit.

## Future Improvements
-   **Expand Data Corpus:** Integrate more diverse and extensive medical texts (e.g., medical journals, clinical guidelines, more specialized textbooks) to improve knowledge breadth and depth.
-   **Extended Training:** Increase the number of training epochs (`num_train_epochs` in `train_slm.py`) to allow the model to learn more effectively from the data.
-   **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and other training arguments to optimize model performance.
-   **Evaluation Metrics:** Implement more robust evaluation metrics (e.g., perplexity, ROUGE scores for summarization tasks) to quantitatively assess model quality.
-   **Model Architecture:** Explore other small language model architectures or different pre-trained models.