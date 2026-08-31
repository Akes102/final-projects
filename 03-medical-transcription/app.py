
import json
import os
from pathlib import Path

import gradio as gr
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
# STRUCTURED OUTPUT SCHEMA
# ============================================================

MEDICAL_SCHEMA = {
    "type": "object",
    "properties": {
        "patient_name": {
            "type": "string"
        },
        "date": {
            "type": "string"
        },
        "chief_complaint": {
            "type": "string"
        },
        "symptoms": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "medications": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "follow_up": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "summary": {
            "type": "string"
        }
    },
    "required": [
        "patient_name",
        "date",
        "chief_complaint",
        "symptoms",
        "medications",
        "diagnoses",
        "follow_up",
        "summary"
    ],
    "additionalProperties": False
}


# ============================================================
# ORGANIZE TRANSCRIPTION
# ============================================================

def organize_transcription(transcription):

    if not transcription.strip():

        return (
            "Paste a transcription first.",
            ""
        )

    prompt = f"""
You are organizing a medical transcription.

Extract information into the requested structured fields.

IMPORTANT:

- Do not invent information.
- Do not diagnose the patient.
- Preserve uncertainty.
- If information does not appear in the transcription,
  return an empty string or empty list.
- Keep the information faithful to the original text.

TRANSCRIPTION:

{transcription}
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You organize medical text into structured "
                        "information for administrative review. "
                        "You do not provide medical diagnosis or treatment."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0,

            max_tokens=1500,

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "medical_transcription",
                    "strict": True,
                    "schema": MEDICAL_SCHEMA
                }
            }
        )

        raw_response = response.choices[0].message.content

        data = json.loads(raw_response)

        formatted_json = json.dumps(
            data,
            indent=4
        )

        return (
            formatted_json,
            data["summary"]
        )

    except Exception as error:

        return (
            f"Error:\n\n{error}",
            ""
        )


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="Medical Transcription Organizer"
) as app:

    gr.Markdown(
        """
        # 🏥 Medical Transcription Organizer

        Convert unstructured transcription into structured information.

        **Educational prototype only. Use synthetic or de-identified
        information. Do not upload real patient information.**
        """
    )

    transcription = gr.Textbox(
        label="Medical Transcription",
        placeholder=(
            "Example:\n"
            "Patient attended the clinic with a headache..."
        ),
        lines=15
    )

    organize_button = gr.Button(
        "Organize Transcription",
        variant="primary"
    )

    structured_output = gr.Code(
        label="Structured Information",
        language="json"
    )

    summary_output = gr.Markdown(
        label="Summary"
    )

    organize_button.click(
        organize_transcription,

        inputs=[
            transcription
        ],

        outputs=[
            structured_output,
            summary_output
        ]
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Medical Transcription Organizer")
    print("=" * 50)

    app.launch()
