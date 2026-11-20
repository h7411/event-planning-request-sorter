"""Your code. One stub per milestone; build them in order and keep each one small."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from app.data import (
    CATEGORIES,
    CATEGORY_TOKENS,
    RAW_DATASET,
    get_categories,
    get_category_counts,
    get_dataset,
)


# ==============================================================================
# Milestone 1: Define common event request types and dataset
# ==============================================================================

def milestone_1() -> dict[str, Any]:
    """Define common event request categories and create a balanced dataset."""
    categories = get_categories()
    dataset = get_dataset()
    category_counts = get_category_counts()

    return {
        "status": "completed",
        "categories": categories,
        "num_categories": len(categories),
        "total_examples": len(dataset),
        "category_counts": category_counts,
        "dataset": dataset,
    }


# ==============================================================================
# Milestone 2: Configure GPT-2 with extended tokenizer and model
# ==============================================================================

def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# ==============================================================================
# Milestone 3: Train model using Hugging Face Trainer
# ==============================================================================

def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# ==============================================================================
# Milestone 4: Inference CLI interface
# ==============================================================================

def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# ==============================================================================
# Milestone 5: Save classified requests to file
# ==============================================================================

def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
