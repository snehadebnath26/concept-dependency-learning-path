import json
import re
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b-instruct"


def _candidate_dict(candidate):
    if isinstance(candidate, dict):
        return candidate
    if hasattr(candidate, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(candidate)
    return {
        "source": candidate.source,
        "target": candidate.target,
        "reasons": getattr(candidate, "reasons", [])
    }


def _normalise(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _concept_terms(concept, concept_id):
    terms = [concept_id.replace("_", " "), concept.get("name", "")]
    terms.extend(concept.get("aliases", []))

    if concept_id == "recursion":
        terms.extend(["recursive", "recursively", "recursive traversal"])

    return list({
        _normalise(term)
        for term in terms
        if isinstance(term, str) and len(_normalise(term)) >= 3
    })


def _mentions_concept(text, terms):
    normalised = f" {_normalise(text)} "
    return any(f" {term} " in normalised for term in terms)


def _supports_direction(source, target, quote):
    text = _normalise(quote)
    source = _normalise(source)
    target = _normalise(target)

    if not source or not target:
        return False

    source_base = source[:-1] if source.endswith("s") else source
    target_base = target[:-1] if target.endswith("s") else target

    patterns = [
        rf"\b{re.escape(target)}\b.*\b(?:is a|is an|kind of|type of)\b.*\b{re.escape(source)}\b",
        rf"\b{re.escape(source)}\b.*\b(?:prerequisite for|required for|needed for|basis for)\b.*\b{re.escape(target)}\b",
        rf"\b{re.escape(target)}\b.*\b(?:requires|depends on|builds on)\b.*\b{re.escape(source)}\b",
        rf"\b{re.escape(target)}\b.*\b(?:defined|implemented|performed|carried out)\b.*\b(?:using|with|through|by)\b.*\b{re.escape(source)}\b",
        rf"\b{re.escape(target)}\b.*\b(?:defined|implemented|performed|carried out)\b.*\brecursively\b",
        rf"\b{re.escape(target)}\b.*\b(?:traversals?|operations?)\b.*\b(?:defined|performed|implemented)\b.*\brecursively\b",
        rf"\b{re.escape(target_base)}s?\b.*\bgeneralises?\b.*\b{re.escape(source_base)}s?\b"
    ]

    if any(re.search(pattern, text) for pattern in patterns):
        return True

    if source in {"recursion", "recursive"}:
        if re.search(r"\brecursiv\w*\b", text):
            return True

    return False


def _request_model(prompt, model=DEFAULT_MODEL):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0}
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot reach Ollama. Make sure Ollama is running."
        ) from exc

    return result["response"]


def _parse_response(response):
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("The model did not return valid JSON.")
        data = json.loads(match.group(0))

    relationships = data.get("relationships")

    if not isinstance(relationships, list):
        raise ValueError("Model output must contain a relationships list.")

    return relationships


def discover_relationships(
    concepts_data,
    chunks,
    candidates,
    prompt_path=None,
    model=DEFAULT_MODEL
):
    if prompt_path is None:
        prompt_path = Path(__file__).parent / "prompts" / "discover_v1.txt"

    instructions = Path(prompt_path).read_text()
    concepts = concepts_data.get("concepts", [])

    concept_map = {
        concept["id"]: concept
        for concept in concepts
        if "id" in concept
    }

    chunk_map = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
        if "chunk_id" in chunk
    }

    candidates = [_candidate_dict(item) for item in candidates]
    discovered = []
    seen = set()

    for candidate in candidates:
        source_id = candidate.get("source")
        target_id = candidate.get("target")

        if not isinstance(source_id, str) or not isinstance(target_id, str):
            continue

        if source_id not in concept_map or target_id not in concept_map:
            continue

        if source_id == target_id or (source_id, target_id) in seen:
            continue

        source_concept = concept_map[source_id]
        target_concept = concept_map[target_id]

        source_terms = _concept_terms(source_concept, source_id)
        target_terms = _concept_terms(target_concept, target_id)

        source_chunk_ids = {
            item.get("chunk_id")
            for item in source_concept.get("sources", [])
            if item.get("chunk_id")
        }

        target_chunk_ids = {
            item.get("chunk_id")
            for item in target_concept.get("sources", [])
            if item.get("chunk_id")
        }

        relevant_chunks = [
            chunk for chunk in chunks
            if (
                _mentions_concept(chunk.get("text", ""), source_terms)
                and _mentions_concept(chunk.get("text", ""), target_terms)
            )
            or chunk.get("chunk_id") in source_chunk_ids
            or chunk.get("chunk_id") in target_chunk_ids
        ]

        if not relevant_chunks:
            continue

        input_data = {
            "source_concept": source_concept,
            "target_concept": target_concept,
            "candidate_reasons": candidate.get("reasons", []),
            "chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc": chunk.get("doc"),
                    "page": chunk.get("page"),
                    "text": chunk.get("text", "")
                }
                for chunk in relevant_chunks
            ]
        }

        prompt = (
            instructions
            + "\n\nEvaluate only the directed candidate pair supplied below. "
            + "The source must be the prerequisite and the target must depend "
            + "on it. Do not reverse the direction. A chunk mentioning both "
            + "concepts is not sufficient by itself. Return an empty "
            + "relationships list if the evidence does not support this "
            + "direction. Do not infer direct prerequisites from transitive "
            + "relationships. Use only exact supplied quotes and chunk IDs. "
            + "Definition chunks for both concepts may be included. "
            + "For example, 'a binary search tree is a binary tree' supports "
            + "binary_tree -> binary_search_tree, not the reverse. "
            + "An AVL tree defined as a self-balancing binary search tree "
            + "supports binary_search_tree -> avl_tree. "
            + "Check grammatical direction and direct dependency carefully.\n\n"
            + json.dumps(input_data, ensure_ascii=False)
        )

        response = _request_model(prompt, model)

        if source_id == "recursion" and target_id == "binary_search_tree":
            print("\nDEBUG: recursion -> binary_search_tree")
            print(response)

        relations = _parse_response(response)

        for relation in relations:
            if not isinstance(relation, dict):
                continue
            if relation.get("source") != source_id:
                continue
            if relation.get("target") != target_id:
                continue
            if relation.get("type", "PREREQUISITE") != "PREREQUISITE":
                continue

            strength = relation.get("strength")
            confidence = relation.get("confidence")
            evidence = relation.get("evidence")

            if strength not in {"hard", "soft"}:
                continue

            if isinstance(confidence, bool) or not isinstance(
                confidence, (int, float)
            ):
                continue

            if not 0 <= confidence <= 1:
                continue

            if not isinstance(evidence, list) or not evidence:
                continue

            verified_evidence = []
            relevant_chunk_ids = {
                entry["chunk_id"] for entry in relevant_chunks
            }

            for item in evidence:
                if not isinstance(item, dict):
                    continue

                chunk_id = item.get("chunk_id")
                quote = item.get("quote")
                chunk = chunk_map.get(chunk_id)

                if not chunk or not isinstance(quote, str) or not quote.strip():
                    continue

                if quote not in chunk.get("text", ""):
                    continue

                if chunk_id not in relevant_chunk_ids:
                    continue

                mentions_source = _mentions_concept(quote, source_terms)
                mentions_target = _mentions_concept(quote, target_terms)

                supports_direction = _supports_direction(
                    source_id.replace("_", " "),
                    target_id.replace("_", " "),
                    quote
                )

                if not supports_direction:
                    continue

                if not (
                    (mentions_source and mentions_target)
                    or chunk_id in target_chunk_ids
                ):
                    continue

                verified = {
                    "chunk_id": chunk_id,
                    "quote": quote
                }

                if chunk.get("doc") is not None:
                    verified["doc"] = chunk["doc"]

                if chunk.get("page") is not None:
                    verified["page"] = chunk["page"]

                if verified not in verified_evidence:
                    verified_evidence.append(verified)

            if not verified_evidence:
                continue

            seen.add((source_id, target_id))

            discovered.append({
                "id": f"r{len(discovered) + 1}",
                "source": source_id,
                "target": target_id,
                "type": "PREREQUISITE",
                "strength": strength,
                "confidence": float(confidence),
                "evidence": verified_evidence,
                "method": "ollama_qwen_discovery_v5"
            })

            break

    return {
        "schema_version": "1.0",
        "course_id": concepts_data.get("course_id", "ds101"),
        "relationships": discovered
    }
