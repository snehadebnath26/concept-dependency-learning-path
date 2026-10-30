import argparse
import json
from pathlib import Path

from cdlp.shared.io import EXAMPLES, read_json, write_json
from cdlp.discovery.candidates import generate_candidates, load_chunks, save_candidates
from cdlp.discovery.llm_discovery import DEFAULT_MODEL, discover_relationships


def main():
    parser = argparse.ArgumentParser(
        description="Discover prerequisite relationships from concepts and chunks."
    )
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    concepts = read_json(args.concepts)
    chunks = load_chunks(args.chunks)

    candidates = generate_candidates(concepts, chunks)
    save_candidates(candidates, args.out_dir / "candidates.json")

    try:
        relationships = discover_relationships(
            concepts,
            chunks,
            candidates,
            model=args.model,
        )
        if not relationships.get("relationships"):
            print(
                "Warning: no relationships were discovered. "
                "Using the example relationships as a fallback."
            )
            relationships = read_json(EXAMPLES / "relationships.json")
        else:
            print(
                f"Discovered {len(relationships['relationships'])} "
                "prerequisite relationships."
            )
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Warning: LLM discovery could not complete: {exc}")
        print("Using example relationships as a fallback.")
        relationships = read_json(EXAMPLES / "relationships.json")

    relationships["schema_version"] = relationships.get("schema_version", "1.0")
    relationships["course_id"] = concepts.get(
        "course_id",
        relationships.get("course_id", "ds101"),
    )

    write_json(args.out_dir / "relationships.json", relationships)
    write_json(
        args.out_dir / "quiz_items.json",
        read_json(EXAMPLES / "quiz_items.json"),
    )
    write_json(
        args.out_dir / "sim_students.json",
        read_json(EXAMPLES / "sim_students.json"),
    )

    print(f"Candidates generated: {len(candidates)}")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
