
import os
from pathlib import Path

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

PROJECTS_DIR = PROJECT_DIR.parent

KNOWLEDGE_BASE = (
    PROJECT_DIR
    / "knowledge-base"
)

load_dotenv(
    PROJECTS_DIR / ".env"
)

API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

if not API_KEY:

    raise ValueError(
        f"OPENAI_API_KEY not found in "
        f"{PROJECTS_DIR / '.env'}"
    )

client = OpenAI(
    api_key=API_KEY
)

MODEL = "gpt-4o-mini"

EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_documents():

    documents = []

    for file_path in KNOWLEDGE_BASE.rglob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        # Simple chunking:
        # paragraphs become chunks
        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        for index, paragraph in enumerate(
            paragraphs
        ):

            documents.append(
                {
                    "source": file_path.name,
                    "chunk": index,
                    "text": paragraph
                }
            )

    return documents


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(texts):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return np.array(
        [
            item.embedding
            for item in response.data
        ],
        dtype=np.float32
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    vectors,
    query_vector
):

    vector_norms = np.linalg.norm(
        vectors,
        axis=1
    )

    query_norm = np.linalg.norm(
        query_vector
    )

    scores = (
        vectors @ query_vector
    ) / (
        vector_norms
        * query_norm
        + 1e-8
    )

    return scores


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

documents = load_documents()

if documents:

    document_vectors = create_embeddings(
        [
            document["text"]
            for document in documents
        ]
    )

else:

    document_vectors = np.empty(
        (0, 0),
        dtype=np.float32
    )


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

def answer_question(
    question,
    number_of_chunks
):

    if not question.strip():

        return (
            "Enter a question.",
            ""
        )

    if not documents:

        return (
            "The knowledge base is empty.",
            ""
        )

    # --------------------------------------------------------
    # Convert question into embedding
    # --------------------------------------------------------

    query_vector = create_embeddings(
        [question]
    )[0]

    # --------------------------------------------------------
    # Compare question against documents
    # --------------------------------------------------------

    scores = cosine_similarity(
        document_vectors,
        query_vector
    )

    # --------------------------------------------------------
    # Find most relevant chunks
    # --------------------------------------------------------

    top_indices = np.argsort(
        scores
    )[::-1][
        :int(number_of_chunks)
    ]

    retrieved_chunks = []

    for index in top_indices:

        document = documents[
            int(index)
        ]

        retrieved_chunks.append(
            {
                "source": document["source"],
                "chunk": document["chunk"],
                "score": float(scores[index]),
                "text": document["text"]
            }
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"""
SOURCE: {chunk['source']}
CHUNK: {chunk['chunk']}
RELEVANCE: {chunk['score']:.3f}

{chunk['text']}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Ask OpenAI using retrieved context
    # --------------------------------------------------------

    prompt = f"""
Answer the user's question using ONLY the supplied
technical documentation.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer cannot be found in the context,
   say that the knowledge base does not contain
   enough information.
4. Give a concise answer.
5. Mention the relevant source when appropriate.

TECHNICAL DOCUMENTATION:

{context}

USER QUESTION:

{question}
"""

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technical documentation "
                    "assistant. Ground your answers in "
                    "the supplied documentation."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0,

        max_tokens=700
    )

    answer = response.choices[0].message.content

    # --------------------------------------------------------
    # Show retrieved sources
    # --------------------------------------------------------

    source_text = "## Retrieved Sources\n\n"

    for chunk in retrieved_chunks:

        source_text += (
            f"**{chunk['source']}** "
            f"(chunk {chunk['chunk']}, "
            f"score {chunk['score']:.3f})\n\n"
            f"{chunk['text']}\n\n"
        )

    return (
        answer,
        source_text
    )


# ============================================================
# GRADIO
# ============================================================

with gr.Blocks(
    title="Technical Documentation RAG"
) as app:

    gr.Markdown(
        """
        # 📚 Technical Documentation RAG Assistant

        Ask questions about the technical documentation
        stored in the local knowledge base.
        """
    )

    question = gr.Textbox(
        label="Question",
        placeholder=(
            "Example: How do I reset a forgotten password?"
        ),
        lines=3
    )

    number_of_chunks = gr.Slider(
        minimum=1,
        maximum=5,
        value=3,
        step=1,
        label="Retrieved Documentation Chunks"
    )

    ask_button = gr.Button(
        "Ask",
        variant="primary"
    )

    answer = gr.Markdown(
        label="Answer"
    )

    sources = gr.Markdown(
        label="Sources"
    )

    ask_button.click(
        answer_question,

        inputs=[
            question,
            number_of_chunks
        ],

        outputs=[
            answer,
            sources
        ]
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Technical Documentation RAG")
    print("=" * 50)

    print(
        f"Documents loaded: {len(documents)}"
    )

    app.launch()