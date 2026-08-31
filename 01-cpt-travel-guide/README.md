# Cape Town AI Travel Guide

An AI-powered travel assistant built with Python, OpenAI and Gradio.

## Purpose

This project demonstrates how to use the OpenAI API to build a conversational application.

The original exercise focused on planning trips to Paris. This version extends the concept into a reusable travel assistant that can answer travel questions and maintain conversation context.

## Technologies

* Python
* OpenAI API
* Gradio
* python-dotenv

## Concepts demonstrated

### 1. OpenAI API

The application sends messages to an OpenAI model and receives generated responses.

### 2. System prompts

The system prompt defines the AI's role and behavior.

### 3. Conversation history

The application stores user and assistant messages and sends them back to the model with each request.

This allows follow-up questions such as:

> Plan me a 3-day trip to Cape Town.

followed by:

> Make it cheaper.

The model can understand that "it" refers to the previously created itinerary.

### 4. Environment variables

The API key is stored in `.env` instead of being hard-coded into the Python source code.

### 5. Gradio

Gradio provides the web interface for interacting with the travel assistant.

## Application architecture

```text
User
 |
 v
Gradio Interface
 |
 v
Conversation History
 |
 v
OpenAI API
 |
 v
AI Response
 |
 v
Conversation History
 |
 v
Gradio Interface
```

## Running the application

From the project directory:

```powershell
python app.py
```

The application will provide a local Gradio URL.

## Example questions

```text
Plan me a 3-day trip to Cape Town.
```

```text
Make it budget friendly.
```

```text
Make it suitable for a family.
```

```text
Remove the most expensive activity.
```

## Important limitation

An LLM should not replace specialized services for live information.

For example, exact driving distance should come from a mapping or routing service.

Current flight availability should come from a flight service.

Current hotel availability should come from a booking service.

A production version could combine these services with OpenAI.

## Possible future improvements

* Destination selection
* Budget input
* Number of travelers
* Trip duration
* Interest selection
* Weather API
* Maps API
* Flight API
* Hotel API
* Restaurant recommendations
* Export itinerary to PDF
* Save trips
* User accounts
* RAG-based travel knowledge
