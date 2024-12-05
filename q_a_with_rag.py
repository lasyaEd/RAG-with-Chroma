# -*- coding: utf-8 -*-
"""
q-a-with-rag.py

This script demonstrates how to implement a Retrieval-Augmented Generation (RAG) system using
the Gemini API and ChromaDB for building and querying an embedding database.

Steps:
1. Indexing
2. Retrieval
3. Generation

Licensed under the Apache License, Version 2.0. For details, see:
https://www.apache.org/licenses/LICENSE-2.0
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry


# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the Gemini API with your API key
genai.configure(api_key=GOOGLE_API_KEY)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for generating embeddings with the Gemini API.
    Supports two modes: 'retrieval_document' for documents and 'retrieval_query' for queries.
    """

    document_mode = True  # Set default mode to document embedding

    def __call__(self, input: Documents) -> Embeddings:
        embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}

        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=embedding_task,
            request_options=retry_policy,
        )
        return response["embedding"]


# Initialize the Chroma database client and embedding function
DB_NAME = "googlecardb"
embed_fn = GeminiEmbeddingFunction()
chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

# Populate the database with documents from the 'data' folder
documents_folder = "data"
documents = []
doc_ids = []

# Loop through .txt files and add them to the database
for idx, filename in enumerate(os.listdir(documents_folder)):
    if filename.endswith(".txt"):
        file_path = os.path.join(documents_folder, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            documents.append(file.read())
            doc_ids.append(str(idx))

if documents:
    db.add(documents=documents, ids=doc_ids)
    print(f"Added {len(documents)} documents to the database.")
else:
    print("No documents found in the folder.")

# Verify the database content
print(f"Database contains {db.count()} documents.")
print("Sample document:", db.peek(1))

# Switch to query mode for retrieval
embed_fn.document_mode = False

# Query the database
query = "Does Chick-fil-A's Terms and Conditions allow class action lawsuits?"
result = db.query(query_texts=[query], n_results=1)

if result["documents"]:
    [[passage]] = result["documents"]
    print("Retrieved passage:", passage)

    # Prepare a prompt for augmented generation
    passage_oneline = passage.replace("\n", " ")
    query_oneline = query.replace("\n", " ")
    prompt = f"""You are a knowledgeable assistant.
    - You can provide answers based on terms and conditions you have access to.
    - If a question is outside your RAG system, you can answer using general knowledge.

    QUESTION: {query_oneline}
    PASSAGE: {passage_oneline}
    """
    print("Prompt:\n", prompt)

    # Generate an answer using the Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    answer = model.generate_content(prompt)
    print("Generated Answer:\n", answer.text)
else:
    print("No relevant passages found in the database.")

"""
Next Steps:
1. Learn more about embeddings and the Gemini API:
   - Intro to embeddings: https://ai.google.dev/gemini-api/docs/embeddings
   - Embeddings chapter: https://developers.google.com/machine-learning/crash-course/embeddings
2. Explore hosted RAG systems:
   - Semantic Retrieval service: https://ai.google.dev/gemini-api/docs/semantic_retrieval
"""