
# MLFF_QD Expert Chatbot 🤖

A RAG (Retrieval-Augmented Generation) chatbot designed to answer questions about MLFF_QD, configurations, and workflows.

## 🌼 Features

- **Hybrid Search:** Combines ChromaDB (Vector) and BM25 (Keyword) for robust retrieval.
- **Re-Ranking:** Uses FlashRank to re-order retrieved documents for higher relevance.
- **LLM Integration:** Powered by Groq (Llama 3) for fast and accurate responses.
- **Interactive UI:** Built with Chainlit for a chat-like experience.
- **Sources Citation:** Displays the specific files used to generate the answer.

## 📁 Project Structure

```text
data/                  # Place your source documents here
│   manual_user_guide.txt
│   paper.docx
│   qa_pairs_new.txt
app_chainlitkeywordRank.py   # Main application logic
config.py             # Configuration settings
.env                  # Environment variables (API Keys)
requirements.txt      # Python dependencies
```

## 🛠 Installation

### Prerequisites
- Anaconda or Miniconda  
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/nlesc-nano/MLFF_QD_ChatBot
cd MLFF_QD_ChatBot
```

### 2. Create a Conda Environment
```bash
conda create -n mlff_bot python=3.11 -y
conda activate mlff_bot
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
1. Create a file named `.env` in the root directory.  
2. Add your Groq API Key:

```bash
GROQ_API_KEY=gsk_your_actual_api_key_here
```

### 5. Prepare Data

Ensure your data files are in the `data/` folder. The app expects:

- `manual_user_guide.txt`
- `qa_pairs_new.txt`
- `paper.docx`

(You can change these filenames in `config.py` if needed.)
#### ⚠️ Important Note on File Formats

The current code supports `.txt` and `.docx` files only.  
If you want to use other file formats (e.g., PDF, CSV), you must modify the `setup_retriever`  
function inside **app_chainlitkeywordRank.py** to include the appropriate document loader.

You can find the list of supported loaders here:  
👉 **[LangChain Document Loaders Documentation](https://docs.langchain.com/oss/python/integrations/document_loaders)**


## 🚀 Usage

Run the Chainlit application:

```bash
chainlit run app_chainlitkeywordRank.py -w
```

The `-w` flag enables auto-reloading when you change code.