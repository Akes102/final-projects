
import json
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

# Project structure:
#
# projects/
# ├── .env
# ├── .venv/
# └── 02-educational-quiz-bot/
#     └── app.py
#
# Go up two levels from app.py to find the shared projects folder.

PROJECTS_DIR = Path(__file__).resolve().parent.parent

load_dotenv(PROJECTS_DIR / ".env")


# ============================================================
# OPENAI
# ============================================================

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY was not found.\n"
        f"Add your API key to: {PROJECTS_DIR / '.env'}"
    )

client = OpenAI(api_key=API_KEY)

MODEL = "gpt-4o-mini"


# ============================================================
# QUIZ JSON SCHEMA
# ============================================================

QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string"
                    },
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "correct_answer": {
                        "type": "string"
                    },
                    "explanation": {
                        "type": "string"
                    }
                },
                "required": [
                    "question",
                    "options",
                    "correct_answer",
                    "explanation"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "questions"
    ],
    "additionalProperties": False
}


# ============================================================
# GENERATE QUIZ
# ============================================================

def generate_quiz(topic, difficulty, number_of_questions):

    if not topic.strip():
        return (
            "Enter a topic first.",
            None,
            ""
        )

    number_of_questions = int(number_of_questions)

    prompt = f"""
Create a multiple-choice educational quiz.

Topic:
{topic}

Difficulty:
{difficulty}

Number of questions:
{number_of_questions}

Requirements:

1. Create exactly {number_of_questions} questions.
2. Each question must have exactly four answer options.
3. Only one option can be correct.
4. The correct_answer must exactly match one of the options.
5. Give a short explanation for the correct answer.
6. Make the questions appropriate for the selected difficulty.
7. Avoid duplicate questions.
8. Return only the requested structured data.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert educational quiz creator. "
                        "Create accurate, clear and useful quizzes."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_tokens=2000,

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "educational_quiz",
                    "strict": True,
                    "schema": QUIZ_SCHEMA
                }
            }
        )

        # Get the model's JSON response
        raw_response = response.choices[0].message.content

        # Convert JSON string into Python dictionary
        quiz_data = json.loads(raw_response)

        # Build a readable quiz
        quiz_lines = []

        for index, question in enumerate(
            quiz_data["questions"],
            start=1
        ):

            quiz_lines.append(
                f"### {index}. {question['question']}"
            )

            for option_number, option in enumerate(
                question["options"],
                start=1
            ):

                letter = chr(64 + option_number)

                quiz_lines.append(
                    f"{letter}. {option}"
                )

            quiz_lines.append("")

        quiz_text = "\n".join(quiz_lines)

        return (
            quiz_text,
            quiz_data,
            "Quiz generated successfully."
        )

    except Exception as error:

        return (
            f"Error generating quiz:\n\n{error}",
            None,
            ""
        )


# ============================================================
# SHOW ANSWERS
# ============================================================

def show_answers(quiz_data):

    if not quiz_data:

        return "Generate a quiz first."

    answer_lines = [
        "## Answer Key"
    ]

    for index, question in enumerate(
        quiz_data["questions"],
        start=1
    ):

        answer_lines.append(
            f"### {index}. {question['correct_answer']}"
        )

        answer_lines.append(
            question["explanation"]
        )

        answer_lines.append("")

    return "\n".join(answer_lines)


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Educational Quiz Bot"
) as app:

    gr.Markdown(
        """
        # 🧠 Educational Quiz Bot

        Generate quizzes about almost any subject.

        Choose a topic, difficulty and number of questions.
        """
    )

    # --------------------------------------------------------
    # Quiz settings
    # --------------------------------------------------------

    with gr.Row():

        topic = gr.Textbox(
            label="Topic",
            placeholder="Example: Python programming"
        )

        difficulty = gr.Dropdown(
            choices=[
                "Beginner",
                "Intermediate",
                "Advanced"
            ],
            value="Beginner",
            label="Difficulty"
        )

        number_of_questions = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Number of Questions"
        )

    # --------------------------------------------------------
    # Buttons
    # --------------------------------------------------------

    generate_button = gr.Button(
        "Generate Quiz",
        variant="primary"
    )

    show_answers_button = gr.Button(
        "Show Answer Key"
    )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    quiz_output = gr.Markdown(
        label="Quiz"
    )

    status = gr.Markdown()

    quiz_state = gr.State()


    answer_output = gr.Markdown(
        label="Answer Key"
    )


    # --------------------------------------------------------
    # Generate quiz
    # --------------------------------------------------------

    generate_button.click(
        generate_quiz,

        inputs=[
            topic,
            difficulty,
            number_of_questions
        ],

        outputs=[
            quiz_output,
            quiz_state,
            status
        ]
    )


    # --------------------------------------------------------
    # Show answers
    # --------------------------------------------------------

    show_answers_button.click(
        show_answers,

        inputs=[
            quiz_state
        ],

        outputs=[
            answer_output
        ]
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Educational Quiz Bot")
    print("=" * 50)
    print(f"Model: {MODEL}")
    print(f"Project directory: {PROJECTS_DIR}")
    print("Starting application...")
    print("=" * 50)

    app.launch()

