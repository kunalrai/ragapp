import os
import faiss
import docx
import openai
import numpy as np
import pandas as pd
from typing import List, Tuple
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from openai import AzureOpenAI
import re

# Load environment variables
load_dotenv()

# Set up Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE")
)

app = Flask(__name__)
vector_dimension = 1536
index = None
documents = []

# Chunking with sentence logic
def chunk_text(content: str, n_sentences: int = 5) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', content)
    return [" ".join(sentences[i:i + n_sentences]) for i in range(0, len(sentences), n_sentences)]

def read_text_file(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    return chunk_text(content)

def read_docx_file(path: str) -> List[str]:
    doc = docx.Document(path)
    content = " ".join(p.text for p in doc.paragraphs)
    return chunk_text(content)

def read_excel_file(path: str) -> List[str]:
    df = pd.read_excel(path)
    content = df.to_string()
    return chunk_text(content)

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=[text],
        model=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENGINE")
    )
    return response.data[0].embedding
def get_embeddings(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        input=texts,
        model=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENGINE")
    )
    return [item.embedding for item in response.data]


def get_embeddings(texts: List[str]) -> List[List[float]]:
    return [get_embedding(t) for t in texts]

def initialize_faiss_index():
    global index, documents
    index = faiss.IndexFlatL2(vector_dimension)
    docs_dir = os.path.join(os.path.dirname(__file__), 'docs')
    if not os.path.exists(docs_dir):
        return

    for filename in os.listdir(docs_dir):
        path = os.path.join(docs_dir, filename)
        chunks = []
        if filename.endswith('.txt'):
            chunks = read_text_file(path)
        elif filename.endswith('.docx'):
            chunks = read_docx_file(path)
        elif filename.endswith('.xlsx'):
            chunks = read_excel_file(path)

        if chunks:
            embeddings = get_embeddings(chunks)
            index.add(np.array(embeddings).astype('float32'))
            documents.extend(chunks)

def search_similar_chunks(query: str, k: int = 5, threshold: float = 0.7) -> List[Tuple[str, float]]:
    query_vector = get_embedding(query)
    distances, indices = index.search(np.array([query_vector]).astype('float32'), k)
    results = [(documents[i], float(d)) for i, d in zip(indices[0], distances[0]) if d < threshold]
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    try:
        query = request.form['query']
        similar_chunks = search_similar_chunks(query)

        if not similar_chunks:
            return jsonify({"answer": "No relevant information found in documents."})

        context = "\n---\n".join(chunk[0] for chunk in similar_chunks)
        print("\n===== CONTEXT SENT TO MODEL =====\n", context[:500])

        messages = [
            {"role": "system", "content": "You are a domain expert answering based on the given context. If unsure, say 'Not found in docs'."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_ENGINE"),
            messages=messages,
            temperature=0.5,
            max_tokens=500
        )

        return jsonify({"answer": response.choices[0].message.content})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test_chat():
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"}
        ]
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_ENGINE"),
            messages=messages,
            temperature=0.7,
            max_tokens=50
        )
        return jsonify({"answer": response.choices[0].message.content})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    initialize_faiss_index()
    app.run(debug=True)