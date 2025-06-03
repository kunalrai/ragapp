# RAG (Retrieval-Augmented Generation) Web Application

A Flask-based web application that implements Retrieval-Augmented Generation using Azure OpenAI and FAISS for efficient document search and question answering.

## Features

- Document processing for multiple file types:
  - Text files (.txt)
  - Word documents (.docx)
  - Excel spreadsheets (.xlsx)
- Vector similarity search using FAISS
- Azure OpenAI integration for embeddings and chat completion
- Web interface for asking questions about your documents
- Secure configuration using environment variables

## Prerequisites

- Python 3.8+
- Azure OpenAI Service access with:
  - GPT-3.5 Turbo model deployment
  - text-embedding-ada-002 model deployment
- Valid Azure OpenAI API key

## Installation

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/yourusername/ragapp.git
cd ragapp
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file based on `.env.example`:
```
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_BASE=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_ENGINE=gpt-35-turbo
AZURE_OPENAI_EMBEDDINGS_ENGINE=text-embedding-ada-002
```

## Usage

1. Place your documents in the `docs/` folder. Supported formats:
   - `.txt` files
   - `.docx` files
   - `.xlsx` files

2. Run the application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

4. Enter your questions in the web interface to search through your documents and get AI-powered answers.

## How It Works

1. **Document Processing**: The application reads and chunks documents from the `docs/` folder.

2. **Vector Embeddings**: Each text chunk is converted to a vector embedding using Azure OpenAI's text-embedding-ada-002 model.

3. **Similarity Search**: When a question is asked, FAISS finds the most relevant document chunks using vector similarity search.

4. **Answer Generation**: The relevant chunks and the question are sent to Azure OpenAI's GPT-3.5 Turbo model to generate a contextual answer.

## Security Notes

- Never commit your `.env` file to version control
- Keep your Azure OpenAI API key secure
- Use environment variables for sensitive configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.