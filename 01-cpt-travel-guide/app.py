
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

# app.py:
# projects/
# └── 01-cpt-travel-guide/
#     └── app.py
#
# parent.parent = projects/

PROJECTS_DIR = Path(__file__).resolve().parent.parent

load_dotenv(PROJECTS_DIR / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY was not found.\n"
        f"Add it to: {PROJECTS_DIR / '.env'}"
    )

client = OpenAI(api_key=API_KEY)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an intelligent AI travel agent.

Your job is to help users plan trips and make better travel
decisions.

You can help with:

- Destinations
- Attractions
- Itineraries
- Budgets
- Activities
- Transportation
- Accommodation
- Food
- Family-friendly activities
- Outdoor activities
- Hiking
- Travel preparation
- Packing lists

Give practical, concise and organized answers.

When creating an itinerary, consider:

- Number of days
- Number of travelers
- Budget
- Interests
- Transportation
- Travel time
- Food
- Activities

When recommending activities, explain why each recommendation
fits the user's requirements.

Do not pretend to have live information about:

- Traffic
- Flight availability
- Hotel availability
- Restaurant availability
- Current prices
- Exact driving distances
- Weather

unless the user provides the information or an external
service provides it.

If important information is missing, make a reasonable
assumption and clearly state it.

When the user asks for a budget, provide an approximate
estimate and label it as an estimate.

Use headings and bullet points when they improve readability.

Do not invent facts just to provide an answer.
"""


# ============================================================
# CONVERSATION
# ============================================================

def new_conversation():
    """
    Creates a fresh conversation containing the system prompt.
    """

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


# ============================================================
# OPENAI REQUEST
# ============================================================

def get_ai_response(user_message, conversation):
    """
    Sends the conversation to OpenAI and returns the response.
    """

    conversation.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=conversation,
            temperature=0.0,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        conversation.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        return answer

    except Exception:

        # Remove the failed user message
        conversation.pop()

        raise


# ============================================================
# CHAT FUNCTION
# ============================================================

def chat(user_message, history, conversation):
    """
    Handles messages from the Gradio interface.
    """

    if not user_message.strip():

        return (
            "",
            history,
            conversation
        )

    if conversation is None:
        conversation = new_conversation()

    if history is None:
        history = []

    try:

        answer = get_ai_response(
            user_message,
            conversation
        )

    except Exception as error:

        answer = f"Something went wrong:\n\n{error}"

    history = history + [
        {
            "role": "user",
            "content": user_message
        },
        {
            "role": "assistant",
            "content": answer
        }
    ]

    return (
        "",
        history,
        conversation
    )


# ============================================================
# RESET CHAT
# ============================================================

def clear_chat():

    return (
        [],
        new_conversation()
    )


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="Cape Town AI Travel Guide"
) as app:

    gr.Markdown(
        """
        # 🇿🇦 Cape Town AI Travel Guide

        Your AI-powered travel planning assistant.

        Ask about attractions, itineraries, budgets,
        activities, food, transport and more.
        """
    )

    chatbot = gr.Chatbot(
    label="Travel Assistant",
    height=500

    )

    conversation_state = gr.State(
        new_conversation()
    )

    with gr.Row():

        user_input = gr.Textbox(
            label="Ask your travel question",
            placeholder="Example: Plan a 3-day budget trip to Cape Town",
            lines=2,
            scale=8
        )

        send_button = gr.Button(
            "Send",
            variant="primary",
            scale=1
        )

    clear_button = gr.Button(
        "Clear Conversation"
    )

    gr.Markdown(
        """
        ## Try these questions

        **Trip planning**
        > Plan me a 3-day trip to Cape Town.

        **Budget**
        > Make the trip budget friendly.

        **Family**
        > Make it suitable for a family with children.

        **Follow-up**
        > Remove the most expensive activity.

        **Food**
        > Suggest affordable local food.

        **Outdoor activities**
        > Give me three outdoor activities.

        **Packing**
        > What should I pack?

        **Different destination**
        > Plan me a weekend trip to Johannesburg.
        """
    )

    # Send button
    send_button.click(
        chat,
        inputs=[
            user_input,
            chatbot,
            conversation_state
        ],
        outputs=[
            user_input,
            chatbot,
            conversation_state
        ]
    )

    # Enter key
    user_input.submit(
        chat,
        inputs=[
            user_input,
            chatbot,
            conversation_state
        ],
        outputs=[
            user_input,
            chatbot,
            conversation_state
        ]
    )

    # Clear button
    clear_button.click(
        clear_chat,
        outputs=[
            chatbot,
            conversation_state
        ]
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Cape Town AI Travel Guide")
    print("=" * 50)
    print(f"Model: {MODEL}")
    print(f"Project directory: {PROJECTS_DIR}")
    print("Starting application...")
    print("=" * 50)

    app.launch()
