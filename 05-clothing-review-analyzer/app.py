
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans


# ============================================================
# CONFIGURATION
# ============================================================

PROJECTS_DIR = Path(__file__).resolve().parent.parent

load_dotenv(PROJECTS_DIR / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        f"OPENAI_API_KEY not found in {PROJECTS_DIR / '.env'}"
    )

client = OpenAI(
    api_key=API_KEY
)

EMBEDDING_MODEL = "text-embedding-3-small"

MODEL = "gpt-4o-mini"


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(texts):

    embeddings = []

    batch_size = 100

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            start:start + batch_size
        ]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )

        embeddings.extend(
            item.embedding
            for item in response.data
        )

    return np.array(
        embeddings,
        dtype=np.float32
    )


# ============================================================
# ANALYZE REVIEWS
# ============================================================

def analyze_reviews(
    file,
    number_of_clusters
):

    if file is None:

        return (
            "Upload a CSV containing a `review` column.",
            None
        )

    try:

        df = pd.read_csv(file)

        if "review" not in df.columns:

            return (
                "The CSV must contain a `review` column.",
                None
            )

        df = df.dropna(
            subset=["review"]
        ).copy()

        reviews = (
            df["review"]
            .astype(str)
            .tolist()
        )

        if len(reviews) < 2:

            return (
                "At least two reviews are required.",
                df
            )

        # ----------------------------------------------------
        # Convert reviews into embeddings
        # ----------------------------------------------------

        embeddings = create_embeddings(
            reviews
        )

        # ----------------------------------------------------
        # Cluster similar reviews
        # ----------------------------------------------------

        number_of_clusters = min(
            int(number_of_clusters),
            len(reviews)
        )

        model = KMeans(
            n_clusters=number_of_clusters,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(
            embeddings
        )

        df["topic_cluster"] = labels

        # ----------------------------------------------------
        # Collect examples from each cluster
        # ----------------------------------------------------

        cluster_examples = {}

        for cluster in sorted(
            set(labels)
        ):

            examples = (
                df.loc[
                    df["topic_cluster"] == cluster,
                    "review"
                ]
                .head(5)
                .tolist()
            )

            cluster_examples[
                str(cluster)
            ] = examples

        # ----------------------------------------------------
        # Ask OpenAI to name the topics
        # ----------------------------------------------------

        prompt = f"""
Identify the themes represented by these review clusters.

Return JSON.

Each cluster should contain:

- topic
- explanation

Do not invent themes that are not supported by
the review examples.

CLUSTERS:

{json.dumps(
    cluster_examples,
    indent=4
)}
"""

        response = client.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a customer review topic analyst."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0,

            max_tokens=1000
        )

        topics = response.choices[0].message.content

        return (
            topics,
            df
        )

    except Exception as error:

        return (
            f"Error:\n\n{error}",
            None
        )


# ============================================================
# GRADIO
# ============================================================

with gr.Blocks(
    title="Clothing Review Analyzer"
) as app:

    gr.Markdown(
        """
        # 👕 Clothing Review Topic Analyzer

        This application uses embeddings to find reviews
        that have similar meanings.
        """
    )

    csv_file = gr.File(
        label="Reviews CSV",
        file_types=[".csv"],
        type="filepath"
    )

    clusters = gr.Slider(
        minimum=2,
        maximum=8,
        value=4,
        step=1,
        label="Number of Topic Groups"
    )

    analyze_button = gr.Button(
        "Analyze Reviews",
        variant="primary"
    )

    topics_output = gr.Markdown(
        label="Detected Topics"
    )

    dataframe_output = gr.Dataframe(
        label="Clustered Reviews"
    )

    analyze_button.click(
        analyze_reviews,

        inputs=[
            csv_file,
            clusters
        ],

        outputs=[
            topics_output,
            dataframe_output
        ]
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Clothing Review Topic Analyzer")
    print("=" * 50)

    app.launch()
