
import os
from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


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

client = OpenAI(api_key=API_KEY)

MODEL = "gpt-4o-mini"


# ============================================================
# ANALYZE CSV
# ============================================================

def analyze_stock_data(file):

    if file is None:

        return (
            "Upload a CSV file first.",
            None
        )

    try:

        # Read CSV
        df = pd.read_csv(file)

        required_columns = {
            "ticker",
            "company",
            "price",
            "daily_change_pct",
            "volume"
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:

            return (
                "Missing columns:\n\n"
                + "\n".join(sorted(missing_columns)),
                df
            )

        # Convert records to dictionaries
        records = df.to_dict(
            orient="records"
        )

        prompt = f"""
Analyze the following stock market dataset.

IMPORTANT:

This is a supplied data snapshot.

Do not claim the data is live.

Do not provide personalized financial advice.

For each company:

1. Identify whether the daily movement is positive,
   negative or approximately flat.
2. Give one short observation.
3. Identify unusually high trading volume if possible.

Then provide a short overall summary.

DATA:

{records}
"""

        response = client.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial data analyst. "
                        "Analyze only the data supplied by the user."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0,

            max_tokens=1800
        )

        analysis = response.choices[0].message.content

        return (
            analysis,
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
    title="Stock Market Data Enricher"
) as app:

    gr.Markdown(
        """
        # 📈 Stock Market Data Enricher

        Upload a CSV and use AI to turn numerical market
        data into human-readable observations.
        """
    )

    csv_file = gr.File(
        label="Stock CSV",
        file_types=[".csv"],
        type="filepath"
    )

    analyze_button = gr.Button(
        "Analyze Data",
        variant="primary"
    )

    analysis_output = gr.Markdown(
        label="AI Analysis"
    )

    data_output = gr.Dataframe(
        label="Dataset"
    )

    analyze_button.click(
        analyze_stock_data,

        inputs=[
            csv_file
        ],

        outputs=[
            analysis_output,
            data_output
        ]
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Stock Market Data Enricher")
    print("=" * 50)

    app.launch()
