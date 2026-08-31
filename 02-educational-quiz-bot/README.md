````markdown
# Educational Quiz Bot

An AI-powered educational quiz generator built with Python, OpenAI and Gradio.

## Purpose

This project demonstrates how the OpenAI API can generate structured educational content.

The user chooses:

- Topic
- Difficulty
- Number of questions

The application sends those requirements to OpenAI and receives a structured JSON response.

Python then converts the JSON into a readable quiz.

## Technologies

- Python
- OpenAI API
- Gradio
- python-dotenv
- JSON

## Core concept

The application follows this process:

User input
↓
Prompt
↓
OpenAI
↓
Structured JSON
↓
Python
↓
Quiz interface

## Structured output

Instead of asking the model to return ordinary text, the application defines a JSON schema.

Each question contains:

- question
- options
- correct_answer
- explanation

This makes the model's output predictable enough for Python to process.

## Running the application

From the project folder:

```powershell
python app.py
````

Then open the Gradio URL shown in the terminal.

## Test examples

### Python

Topic:

```text
Python programming
```

Difficulty:

```text
Beginner
```

Questions:

```text
5
```

### Networking

Topic:

```text
Computer networking
```

Difficulty:

```text
Intermediate
```

### Artificial Intelligence

Topic:

```text
Machine learning
```

Difficulty:

```text
Advanced
```

## Real-world applications

The same concept can be used to build:

* IT certification practice
* Python learning assistants
* School revision tools
* Interview preparation
* Employee training
* Driving theory practice
* Language learning
* Technical certification preparation

## What this project teaches

1. Dynamic prompts
2. Structured JSON output
3. JSON parsing
4. OpenAI API calls
5. Gradio interfaces
6. Turning AI output into application data
