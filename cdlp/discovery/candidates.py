
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    source: str
    target: str
    reasons: tuple[str, ...]


def normalise(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def phrase_in_text(phrase, text):
    phrase = normalise(phrase)
    text = normalise(text)

    if not phrase:
        return False

    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def load_chunks(path):
    chunks = []

    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def _concept_names(concept):
    return [
        name
        for name in [concept.get("name", ""), *concept.get("aliases", [])]
        if name.strip()
    ]


def _source_position(concept, document_order):
    positions = []

    for source in concept.get("sources", []):
        doc = source.get("doc", "")
        page = source.get("page", 0)

        if doc in document_order:
            positions.append((document_order[doc], page))

    return min(positions) if positions else None


def _embedding_pairs(concepts, threshold=0.45):
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return set(), False

    if len(concepts) < 2:
        return set(), True

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [
            " ".join(
                [
                    concept.get("name", ""),
                    concept.get("description", ""),
                    " ".join(concept.get("aliases", [])),
                ]
            )
            for concept in concepts
        ]

        embeddings = model.encode(texts, normalize_embeddings=True)
        similarities = np.matmul(embeddings, embeddings.T)
    except Exception:
        return set(), False

    pairs = set()

    for i, source in enumerate(concepts):
        for j, target in enumerate(concepts):
            if i != j and similarities[i, j] >= threshold:
                pairs.add((source["id"], target["id"]))

    return pairs, True


def generate_candidates(concepts_data, chunks, threshold=0.45):
    concepts = concepts_data["concepts"]

    documents = sorted(
        {
            chunk.get("doc", "")
            for chunk in chunks
            if chunk.get("doc")
        }
    )

    document_order = {
        doc: index
        for index, doc in enumerate(documents)
    }

    positions = {
        concept["id"]: _source_position(concept, document_order)
        for concept in concepts
    }

    reasons = {}

    def add(source, target, reason):
        if source != target:
            reasons.setdefault((source, target), set()).add(reason)

    embedding_pairs, _ = _embedding_pairs(concepts, threshold)

    for source, target in embedding_pairs:
        add(source, target, "embedding_similarity")

    for chunk in chunks:
        chunk_text = " ".join(
            [
                chunk.get("heading", ""),
                chunk.get("text", ""),
            ]
        )

        present = [
            concept["id"]
            for concept in concepts
            if any(
                phrase_in_text(name, chunk_text)
                for name in _concept_names(concept)
            )
        ]

        for index, source in enumerate(present):
            for target in present[index + 1:]:
                add(source, target, "same_chunk")
                add(target, source, "same_chunk")

    for source in concepts:
        for target in concepts:
            if source["id"] == target["id"]:
                continue

            source_id = source["id"]
            target_id = target["id"]

            source_position = positions.get(source_id)
            target_position = positions.get(target_id)

            if (
                source_position is not None
                and target_position is not None
                and source_position < target_position
            ):
                add(source_id, target_id, "course_order")

            source_names = _concept_names(source)
            target_names = _concept_names(target)

            if any(
                phrase_in_text(name, target.get("description", ""))
                for name in source_names
            ):
                add(source_id, target_id, "name_in_definition")

            if any(
                phrase_in_text(name, source.get("description", ""))
                for name in target_names
            ):
                add(target_id, source_id, "name_in_definition")

    return [
        Candidate(source, target, tuple(sorted(candidate_reasons)))
        for (source, target), candidate_reasons in sorted(reasons.items())
    ]


def save_candidates(candidates, path):
    output = [asdict(candidate) for candidate in candidates]

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")
