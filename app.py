import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import faiss
import docx
import openai
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Azure OpenAI settings
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize FAISS with dimensions for Ada embedding model
vector_dimension = 1536  # Dimension for text-embedding-ada-002
index = None
documents = []

def read_text_file(file_path: str) -> List[str]:
    """Read content from a text file and split into chunks."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return [content[i:i+1000] for i in range(0, len(content), 1000)]

def read_docx_file(file_path: str) -> List[str]:
    """Read content from a .docx file and split into chunks."""
    doc = docx.Document(file_path)
    content = " ".join([paragraph.text for paragraph in doc.paragraphs])
    return [content[i:i+1000] for i in range(0, len(content), 1000)]

def read_excel_file(file_path: str) -> List[str]:
    """Read content from an Excel file and convert to text chunks."""
    df = pd.read_excel(file_path)
    content = df.to_string()
    return [content[i:i+1000] for i in range(0, len(content), 1000)]

def get_embedding(text: str) -> List[float]:
    """Get embedding from Azure OpenAI API."""
    response = openai.Embedding.create(
        input=text,
        engine=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENGINE")
    )
    return response['data'][0]['embedding']

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts."""
    all_embeddings = []
    for text in texts:
        embedding = get_embedding(text)
        all_embeddings.append(embedding)
    return all_embeddings

def initialize_faiss_index():
    """Initialize FAISS index with documents from the docs folder."""
    global index, documents
    
    # Create a new FAISS index
    index = faiss.IndexFlatL2(vector_dimension)
    documents = []
    
    # Process all files in the docs directory
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    if not os.path.exists(docs_dir):
        return
    
    for filename in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, filename)
        chunks = []
        
        if filename.endswith('.txt'):
            chunks = read_text_file(file_path)
        elif filename.endswith('.docx'):
            chunks = read_docx_file(file_path)
        elif filename.endswith('.xlsx'):
            chunks = read_excel_file(file_path)
            
        if chunks:
            # Convert chunks to embeddings and add to index
            embeddings = get_embeddings(chunks)
            index.add(np.array(embeddings).astype('float32'))
            documents.extend(chunks)

def search_similar_chunks(query: str, k: int = 3) -> List[Tuple[str, float]]:
    """Search for similar chunks using FAISS."""
    query_vector = get_embedding(query)
    distances, indices = index.search(np.array([query_vector]).astype('float32'), k)
    results = [(documents[idx], float(distance)) for idx, distance in zip(indices[0], distances[0])]
    return results

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    try:
        query = request.form['query']
        
        # Get similar chunks
        similar_chunks = search_similar_chunks(query)
        context = "\n".join([chunk[0] for chunk in similar_chunks])
        
        # Prepare prompt for Azure OpenAI
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context. If you cannot find the answer in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
          # Get response from Azure OpenAI
        response = openai.ChatCompletion.create(
            engine=os.getenv("AZURE_OPENAI_ENGINE"),
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    initialize_faiss_index()
    app.run(debug=True)
