
import argparse
import json
from pathlib import Path

from cdlp.shared.io import read_json, write_json
from cdlp.shared.llm import DEFAULT_MODEL, chat_json


def load_chunks(path):
    chunks = {}

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunk = json.loads(line)
                chunks[chunk["chunk_id"]] = chunk

    return chunks


def get_chunk_text(chunk):
    for key in ("text", "content", "page_content", "body"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    return ""


def verify_evidence(edge, chunks):
    verified = []

    for item in edge.get("evidence", []):
        if not isinstance(item, dict):
            continue

        chunk = chunks.get(item.get("chunk_id"))
        quote = item.get("quote", "")

        if not chunk or not isinstance(quote, str) or not quote.strip():
            continue

        text = get_chunk_text(chunk)

        if quote.casefold() in text.casefold():
            evidence = dict(item)
            evidence.setdefault("doc", chunk.get("doc", ""))
            evidence.setdefault("page", chunk.get("page", 1))
            verified.append(evidence)

    return verified


def load_necessity(path):
    if not path:
        return {}

    data = read_json(path)
    results = data.get("results", data.get("relationships", []))

    if isinstance(results, dict):
        results = list(results.values())

    lookup = {}

    for item in results if isinstance(results, list) else []:
        if not isinstance(item, dict):
            continue

        source = item.get("source")
        target = item.get("target")

        if source and target:
            lookup[(source, target)] = item

    return lookup


def semantic_check(source, target, evidence, model):
    if not evidence:
        return "none", False, "No verifiable evidence found."

    prompt = (
        "Evaluate whether SOURCE is a prerequisite of TARGET. "
        "A prerequisite means the learner should understand SOURCE "
        "before learning TARGET. Return JSON with rating "
        "(strong, medium, weak, none), supported (boolean), "
        "and reason (string).\n"
        f"SOURCE: {source}\n"
        f"TARGET: {target}\n"
        f"EVIDENCE: {json.dumps(evidence)}"
    )

    result = chat_json(
        prompt,
        system="You are a careful curriculum dependency validator. Reply only in JSON.",
        model=model,
        prompt_version="validation_semantic_v1"
    )

    rating = result.get("rating", "medium")

    if rating not in {"strong", "medium", "weak", "none"}:
        rating = "medium"

    return (
        rating,
        bool(result.get("supported", False)),
        str(result.get("reason", ""))
    )


def path_exists(start, goal, adjacency):
    pending = [start]
    visited = set()

    while pending:
        node = pending.pop()

        if node == goal:
            return True

        if node in visited:
            continue

        visited.add(node)
        pending.extend(adjacency.get(node, []))

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate discovered prerequisite relationships."
    )

    parser.add_argument("--relationships", type=Path, required=True)
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--necessity", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)

    args = parser.parse_args()

    concepts_data = read_json(args.concepts)
    relationships_data = read_json(args.relationships)
    chunks = load_chunks(args.chunks)
    necessity = load_necessity(args.necessity)

    concepts = concepts_data.get("concepts", [])

    names = {
        item["id"]: item.get("name", item["id"])
        for item in concepts
    }

    nodes = [
        {
            "id": item["id"],
            "name": item.get("name", item["id"])
        }
        for item in concepts
    ]

    edges = []
    adjacency = {}
    seen = set()

    for index, relation in enumerate(
        relationships_data.get("relationships", []), 1
    ):
        source = relation.get("source")
        target = relation.get("target")

        if source not in names or target not in names:
            continue

        evidence = verify_evidence(relation, chunks)

        evidence_rating = (
            "strong" if len(evidence) >= 2
            else "medium" if evidence
            else "none"
        )

        semantic_rating, supported, reason = semantic_check(
            names[source],
            names[target],
            evidence,
            args.model
        )

        necessity_item = necessity.get((source, target), {})

        gap = necessity_item.get(
            "necessity_gap",
            necessity_item.get("gap")
        )

        necessity_status = str(
            necessity_item.get("status", "")
        ).upper()

        leakage_rate = necessity_item.get("leakage_rate")
        questions_evaluated = necessity_item.get("questions_evaluated", 0)

        if isinstance(gap, (int, float)):
            necessity_rating = (
                "strong" if gap >= 0.5
                else "medium" if gap >= 0.2
                else "weak" if gap > 0
                else "none"
            )
        else:
            necessity_rating = "none"

        pair = (source, target)

        if source == target:
            graph_status = "self_loop"
        elif pair in seen:
            graph_status = "duplicate"
        elif path_exists(target, source, adjacency):
            graph_status = "cycle"
        else:
            graph_status = "ok"

        if not evidence:
            decision = "REJECT"
        elif graph_status != "ok":
            decision = "REJECT"
        elif not supported or semantic_rating in {"weak", "none"}:
            decision = "HUMAN_REVIEW"
        elif necessity_status != "MEASURED":
            decision = "HUMAN_REVIEW"
        elif not isinstance(gap, (int, float)):
            decision = "HUMAN_REVIEW"
        elif gap <= 0:
            decision = "HUMAN_REVIEW"
        elif questions_evaluated < 2:
            decision = "HUMAN_REVIEW"
        elif not isinstance(leakage_rate, (int, float)):
            decision = "HUMAN_REVIEW"
        elif leakage_rate > 0:
            decision = "HUMAN_REVIEW"
        else:
            decision = "ACCEPT"

        if decision == "ACCEPT":
            adjacency.setdefault(source, []).append(target)

        seen.add(pair)

        edge = {
            "id": str(relation.get("id", f"r{index}")),
            "source": source,
            "target": target,
            "type": "PREREQUISITE",
            "strength": relation.get(
                "strength",
                "hard" if semantic_rating == "strong" else "soft"
            ),
            "decision": decision,
            "confidence": {
                "strong": 0.9,
                "medium": 0.7,
                "weak": 0.4,
                "none": 0.1
            }[semantic_rating],
            "checks": {
                "evidence": evidence_rating,
                "semantic": semantic_rating,
                "necessity": necessity_rating,
                "graph": graph_status
            },
            "necessity_gap": (
                gap if isinstance(gap, (int, float)) else None
            ),
            "evidence": evidence,
            "reason": reason or "Decision based on validation checks."
        }

        edges.append(edge)

    result = {
        "schema_version": relationships_data.get("schema_version", "1.0"),
        "course_id": concepts_data.get(
            "course_id",
            relationships_data.get("course_id", "ds101")
        ),
        "nodes": nodes,
        "edges": edges
    }

    write_json(args.out, result)

    print(f"Nodes: {len(nodes)}")
    print(f"Relationships validated: {len(edges)}")
    print(f"Accepted: {sum(e['decision'] == 'ACCEPT' for e in edges)}")
    print(f"Rejected: {sum(e['decision'] == 'REJECT' for e in edges)}")
    print(f"Human review: {sum(e['decision'] == 'HUMAN_REVIEW' for e in edges)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()

