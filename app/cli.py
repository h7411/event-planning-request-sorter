"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core import (
    CATEGORIES,
    CATEGORY_TOKENS,
    classify_request,
    configure_tokenizer_and_model,
    get_dataset,
    save_classification,
    train_hf_model,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="event-planning-request-sorter",
        description="Event planners use this tool to automatically sort incoming client requests (e.g., \"venue inquiry\", \"catering request\") for efficient task delegation and response prioritization.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. hello command (checked by test_smoke.py)
    sub.add_parser("hello", help="check that everything is wired up")

    # 2. categories command
    sub.add_parser("categories", help="list supported event request categories and custom tokens")

    # 3. train command
    train_parser = sub.add_parser("train", help="fine-tune GPT-2 model on event requests using Hugging Face Trainer")
    train_parser.add_argument("--epochs", type=int, default=2, help="number of training epochs")
    train_parser.add_argument("--output-dir", type=str, default="model", help="directory to save model artifacts")

    # 4. classify command
    classify_parser = sub.add_parser("classify", help="classify an incoming client request")
    classify_parser.add_argument("request", type=str, help="incoming request text (e.g., 'Do you have a hall for 150 guests?')")
    classify_parser.add_argument("--save", type=str, default=None, help="optional path to save classification (e.g., classified.csv)")
    classify_parser.add_argument("--model-dir", type=str, default="model", help="directory containing model artifacts")

    # 5. batch command (stretch idea: process multiple requests from a file)
    batch_parser = sub.add_parser("batch", help="classify multiple requests from an input text file")
    batch_parser.add_argument("--input", type=str, required=True, help="path to text file with one request per line")
    batch_parser.add_argument("--save", type=str, default="batch_classified.csv", help="file to save all classifications")
    batch_parser.add_argument("--model-dir", type=str, default="model", help="directory containing model artifacts")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Event Planning Request Sorter is ready. Open README.md for your milestones.")

    elif args.command == "categories":
        print("\n=== Supported Event Request Categories ===")
        for cat in CATEGORIES:
            token = CATEGORY_TOKENS.get(cat, "")
            print(f"  • {cat.replace('_', ' ').title():<26} Token: {token}")
        print()

    elif args.command == "train":
        print(f"Fine-tuning GPT-2 model on event requests for {args.epochs} epoch(s)...")
        tokenizer, model = configure_tokenizer_and_model()
        dataset = get_dataset()
        res = train_hf_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
        )
        print(f"Training completed successfully! Saved artifacts to: {res['output_dir']}")
        print(f"Epochs: {res['epochs']}, Steps: {res['steps']}, Loss: {res.get('final_loss', 'N/A')}")

    elif args.command == "classify":
        print("\n--- Event Request Classification ---")
        print(f"Request: \"{args.request}\"")

        res = classify_request(args.request, model_dir=args.model_dir)
        pred_cat = res["predicted_category"]
        confidence = res["confidence"]
        token = res["category_token"]

        print(f"Classification: {pred_cat.replace('_', ' ').title()} ({token})")
        print(f"Confidence:     {confidence * 100:.1f}%\n")

        print("Category Probabilities:")
        for cat, score in sorted(res["scores"].items(), key=lambda kv: kv[1], reverse=True):
            bar = "█" * int(score * 20)
            print(f"  {cat:<24} {score * 100:5.1f}%  {bar}")

        if args.save:
            saved_path = save_classification(
                request_text=args.request,
                predicted_category=pred_cat,
                filepath=args.save,
                confidence=confidence,
            )
            print(f"\nSaved classification record to: {saved_path}")

    elif args.command == "batch":
        in_path = Path(args.input)
        if not in_path.exists():
            print(f"Error: input file '{args.input}' not found.")
            sys.exit(1)

        with open(in_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Processing {len(lines)} request(s) from {args.input}...")
        for line in lines:
            res = classify_request(line, model_dir=args.model_dir)
            save_classification(
                request_text=line,
                predicted_category=res["predicted_category"],
                filepath=args.save,
                confidence=res["confidence"],
            )
            print(f"  ✓ {line[:40]:<40} -> {res['predicted_category']} ({res['confidence']*100:.1f}%)")

        print(f"\nAll classifications saved to: {args.save}")


if __name__ == "__main__":
    main()
