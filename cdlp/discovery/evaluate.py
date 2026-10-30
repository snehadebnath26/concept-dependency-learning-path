
import argparse
import csv
import json
from pathlib import Path


def load_gold(path):
    gold = set()

    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            label = row["label"].strip().lower()

            if label in {"hard", "soft"}:
                gold.add((row["source"].strip(), row["target"].strip()))

    return gold


def load_predictions(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    return {
        (item["source"], item["target"])
        for item in data.get("relationships", [])
    }


def evaluate(gold, predictions):
    true_positives = gold & predictions
    false_positives = predictions - gold
    false_negatives = gold - predictions

    precision = (
        len(true_positives) / len(predictions)
        if predictions else 0.0
    )
    recall = (
        len(true_positives) / len(gold)
        if gold else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "gold_edges": len(gold),
        "predicted_edges": len(predictions),
        "true_positives": sorted(true_positives),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate discovered prerequisite relationships."
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/gold/annotations/annotator_b.csv")
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/discovery/relationships.json")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cdlp/discovery/results/evaluation.json")
    )
    args = parser.parse_args()

    gold = load_gold(args.gold)
    predictions = load_predictions(args.predictions)
    results = evaluate(gold, predictions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    print(f"Gold edges: {results['gold_edges']}")
    print(f"Predicted edges: {results['predicted_edges']}")
    print(f"True positives: {len(results['true_positives'])}")
    print(f"False positives: {len(results['false_positives'])}")
    print(f"False negatives: {len(results['false_negatives'])}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    print(f"Results saved to: {args.out}")


if __name__ == "__main__":
    main()
