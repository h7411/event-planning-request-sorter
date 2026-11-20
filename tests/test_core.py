"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass."""
from __future__ import annotations

import csv
from pathlib import Path
import pytest

from app.core import (
    CATEGORIES,
    CATEGORY_TOKENS,
    classify_request,
    configure_tokenizer_and_model,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    save_classification,
)


# ==============================================================================
# Milestone 1: Define common event request types and create dataset
# ==============================================================================

def test_milestone_1():
    result = milestone_1()
    assert result["status"] == "completed"
    assert result["num_categories"] == 5
    assert result["total_examples"] >= 25
    assert "venue_inquiry" in result["categories"]
    assert "catering_request" in result["categories"]
    assert "budget_estimate" in result["categories"]
    assert "schedule_coordination" in result["categories"]
    assert "entertainment_request" in result["categories"]

    # Verify dataset is balanced
    for cat in result["categories"]:
        assert result["category_counts"][cat] >= 5

    # Verify structure of each example
    for example in result["dataset"]:
        assert "text" in example and len(example["text"]) > 10
        assert example["category"] in result["categories"]


# ==============================================================================
# Milestone 2: Configure GPT-2 with extended tokenizer and model
# ==============================================================================

def test_milestone_2():
    result = milestone_2()
    assert result["status"] == "completed"
    assert result["vocab_size_extended"] > result["vocab_size_original"]
    assert result["added_tokens_count"] >= 5
    assert result["pad_token"] is not None

    tokenizer = result["tokenizer"]
    model = result["model"]

    # Verify custom tokens exist in tokenizer
    for cat, token in CATEGORY_TOKENS.items():
        assert token in result["custom_tokens"]

    # Verify tokenizer can encode and decode with custom special tokens
    encoded = tokenizer.encode("<|request|> Test request <|venue_inquiry|>")
    assert isinstance(encoded, list)
    assert len(encoded) > 0

    # Verify model embedding dimension matches extended vocabulary
    assert model.vocab_size == len(tokenizer)


# ==============================================================================
# Milestone 3: Train model on example requests using Hugging Face Trainer
# ==============================================================================

def test_milestone_3(tmp_path):
    output_dir = str(tmp_path / "test_model")
    result = milestone_3(output_dir=output_dir, epochs=1)

    assert result["status"] == "completed"
    assert result["epochs"] == 1
    assert result["trained_samples"] >= 25
    assert result["final_loss"] >= 0.0

    # Verify model and tokenizer saved artifacts exist
    saved_dir = Path(output_dir)
    assert saved_dir.exists()
    assert (saved_dir / "config.json").exists()


# ==============================================================================
# Milestone 4: Inference pipeline to input new requests and show classification
# ==============================================================================

def test_milestone_4():
    # Test catering classification
    catering_result = milestone_4("Can you prepare a 3-course vegan buffet with wine pairings for 100 people?")
    assert catering_result["status"] == "completed"
    assert catering_result["predicted_category"] == "catering_request"
    assert catering_result["confidence"] > 0.3
    assert catering_result["category_token"] == "<|catering_request|>"
    assert len(catering_result["all_scores"]) == 5

    # Test venue classification
    venue_result = milestone_4("Is the grand ballroom available to tour next Monday morning?")
    assert venue_result["predicted_category"] == "venue_inquiry"

    # Test budget classification
    budget_result = milestone_4("Could you send an itemized price estimate and cost quote for the gala?")
    assert budget_result["predicted_category"] == "budget_estimate"

    # Test schedule coordination
    schedule_result = milestone_4("What time does setup begin and what is the curfew timeline for music?")
    assert schedule_result["predicted_category"] == "schedule_coordination"

    # Test entertainment
    ent_result = milestone_4("We want to book a live jazz band and wireless microphones for the stage.")
    assert ent_result["predicted_category"] == "entertainment_request"


# ==============================================================================
# Milestone 5: Allow saving classified requests to a file for review
# ==============================================================================

def test_milestone_5(tmp_path):
    csv_file = str(tmp_path / "classified_requests.csv")
    result = milestone_5(
        request_text="Do you have outdoor patio space available on August 12th?",
        predicted_category="venue_inquiry",
        filepath=csv_file,
        confidence=0.98,
    )

    assert result["status"] == "completed"
    assert result["file_exists"] is True
    assert Path(csv_file).exists()

    # Verify CSV file contents
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 2  # header + 1 record
        header, record = reader[0], reader[1]
        assert "request" in header
        assert "predicted_category" in header
        assert "venue_inquiry" in record
        assert "Do you have outdoor patio space available on August 12th?" in record

    # Append a second record to test history retention
    milestone_5(
        request_text="Can we get a price estimate on linens and chair rentals?",
        predicted_category="budget_estimate",
        filepath=csv_file,
        confidence=0.92,
    )

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 3  # header + 2 records


# ==============================================================================
# CLI and Helper Tests
# ==============================================================================

def test_cli_classify(capsys, tmp_path):
    from app.cli import main

    csv_path = str(tmp_path / "cli_classified.csv")
    main(["classify", "Do you have an open ballroom available on June 10th?", "--save", csv_path])

    captured = capsys.readouterr().out
    assert "Venue Inquiry" in captured
    assert "Confidence:" in captured
    assert Path(csv_path).exists()


def test_cli_categories(capsys):
    from app.cli import main

    main(["categories"])
    captured = capsys.readouterr().out
    assert "Venue Inquiry" in captured
    assert "Catering Request" in captured
    assert "Budget Estimate" in captured
