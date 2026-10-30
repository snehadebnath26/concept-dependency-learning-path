
import csv
from pathlib import Path

from cdlp.discovery.candidates import generate_candidates, load_chunks
from cdlp.shared.io import EXAMPLES, read_json


ROOT = Path(__file__).resolve().parents[3]
GOLD_FILE = ROOT / "data" / "gold" / "annotations" / "annotator_b.csv"


def load_inputs():
    concepts = read_json(EXAMPLES / "concepts.json")
    chunks = load_chunks(EXAMPLES / "chunks.jsonl")
    return concepts, chunks


def load_positive_gold_edges():
    with GOLD_FILE.open(encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        return {
            (row["source"], row["target"])
            for row in rows
            if row["label"].strip().lower() in {"hard", "soft"}
        }


def test_candidates_use_valid_concept_ids():
    concepts, chunks = load_inputs()
    valid_ids = {concept["id"] for concept in concepts["concepts"]}
    candidates = generate_candidates(concepts, chunks)

    assert candidates
    assert all(candidate.source in valid_ids for candidate in candidates)
    assert all(candidate.target in valid_ids for candidate in candidates)
    assert all(candidate.source != candidate.target for candidate in candidates)


def test_candidates_include_filter_reasons():
    concepts, chunks = load_inputs()
    candidates = generate_candidates(concepts, chunks)
    valid_reasons = {
        "embedding_similarity",
        "same_chunk",
        "course_order",
        "name_in_definition",
    }

    assert candidates
    assert all(candidate.reasons for candidate in candidates)
    assert all(
        set(candidate.reasons).issubset(valid_reasons)
        for candidate in candidates
    )


def test_positive_gold_edge_candidate_recall():
    concepts, chunks = load_inputs()
    candidates = generate_candidates(concepts, chunks)
    candidate_edges = {
        (candidate.source, candidate.target)
        for candidate in candidates
    }
    gold_edges = load_positive_gold_edges()

    assert gold_edges
    surviving = gold_edges & candidate_edges
    recall = len(surviving) / len(gold_edges)

    print(
        f"\nCandidate recall: {len(surviving)}/{len(gold_edges)} "
        f"= {recall:.1%}"
    )
    print("Missing gold edges:", sorted(gold_edges - candidate_edges))

    assert recall >= 0.95
