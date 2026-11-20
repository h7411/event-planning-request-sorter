# Event Planning Request Sorter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/k9rz9sk956/badge.svg)](https://ask.thili.ai/b/k9rz9sk956)

This project, 'Event Planning Request Sorter,' provides event planners with a smart tool to streamline their workflow. It leverages a fine-tuned GPT-2 model to automatically categorize client requests (e.g., 'venue inquiry,' 'catering request'), enabling efficient task delegation and response prioritization. This helps event planners quickly understand and act on incoming communications.

## What you are building

**Goal.** You will build a command-line tool that uses a fine-tuned GPT-2 model to automatically classify incoming client requests for event planners, such as "venue inquiry" or "catering request", proving the fine-tuning and inference pipeline logic.

**Done when.** You will have a trained GPT-2 model capable of classifying event planning requests into predefined types. The CLI tool will successfully take a new request, classify it using your model, display the classification, and save the request and its classification to a file.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Define common event request types and create a small dataset of example requests.**  
      Hint: Identify distinct event request categories and create a balanced dataset with example phrases for each.
- [x] **2. Configure GPT-2 with an extended tokenizer and model for custom tokens.**  
      Hint: Extend the GPT-2 tokenizer with special tokens for your custom request types and reconfigure the model accordingly.
- [x] **3. Train the model on the example requests using the Hugging Face Trainer.**  
      Hint: Use the Hugging Face Trainer to fine-tune your GPT-2 model on the prepared dataset of event requests.
- [x] **4. Build an interface (CLI) to input new requests and show their classification.**  
      Hint: Implement an inference pipeline in your CLI to load the fine-tuned model and predict classifications for new user-entered requests.
- [x] **5. Allow saving classified requests to a file for review.**  
      Hint: Add functionality to write the original request and its predicted classification to a text file or CSV for record-keeping.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

### CLI Commands

- **List Categories**: View supported request types and custom tokens:
  ```bash
  python -m app.cli categories
  ```

- **Train Model**: Fine-tune the GPT-2 model using the Hugging Face Trainer:
  ```bash
  python -m app.cli train --epochs 2
  ```

- **Classify Request**: Classify an incoming client request and display confidence score:
  ```bash
  python -m app.cli classify "Can you provide a 3-course vegan buffet for 100 people?"
  ```

- **Save to File**: Save classified requests directly to a CSV or text file for review:
  ```bash
  python -m app.cli classify "Do you have an open ballroom on October 14th?" --save classified_requests.csv
  ```

- **Batch Processing**: Classify multiple requests from a text file:
  ```bash
  python -m app.cli batch --input requests.txt --save batch_classified.csv
  ```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the CLI to process multiple requests from a single input file, outputting all classifications.
- Display a confidence score alongside the predicted classification for each request.
- Add an option to manually correct or re-classify a request before saving it to the file.
