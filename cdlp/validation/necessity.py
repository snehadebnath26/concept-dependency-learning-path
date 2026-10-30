
import argparse
import json
import os
import re
from pathlib import Path

from cdlp.shared.io import read_json, write_json
from cdlp.shared.llm import chat_json


def normalize(value):
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def concept_terms(concept_id, concept):
    terms = [concept_id.replace("_", " "), concept.get("name", "")]
    terms.extend(concept.get("aliases", []))

    target_words = set(
        normalize(concept.get("name", "")).split()
    )
    target_words.update(
        normalize(concept_id.replace("_", " ")).split()
    )

    cleaned = []
    for term in terms:
        if not isinstance(term, str):
            continue

        normalized = normalize(term)
        if not normalized:
            continue

        words = normalized.split()

        if len(words) == 1 and words[0] in target_words:
            continue

        cleaned.append(normalized)

    return cleaned


def contains_term(text, terms):
    normalized_text = f" {normalize(text)} "
    return any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized_text)
        for term in terms
    )


def answer_question(question, target, source, condition, concepts):
    target_concept = concepts[target]
    source_concept = concepts[source]

    context = {
        "target_concept": {
            "id": target,
            "name": target_concept.get("name", target),
            "description": target_concept.get("description", ""),
        },
        "question": question["question"],
        "options": question["options"],
    }

    if condition == "with_prerequisite":
        context["prior_knowledge"] = {
            "id": source,
            "name": source_concept.get("name", source),
            "description": source_concept.get("description", ""),
        }
        instruction = (
            "You have studied the target concept and the stated prerequisite. "
            "Answer the question using this knowledge."
        )
    else:
        context["prior_knowledge"] = {
            "target_concept_only": True,
            "excluded_concept": source,
            "instruction": "Do not assume you have studied this concept.",
        }
        instruction = (
            "You are being tested on the target concept without having studied "
            "the proposed prerequisite. Answer as best you can from the target "
            "concept and the question. Do not deliberately guess the answer "
            "from the excluded prerequisite. Explain your reasoning briefly."
        )

    result = chat_json(
        json.dumps(context, ensure_ascii=False),
        system=(
            "You are a simulated student taking a multiple-choice quiz. "
            + instruction
            + " Return JSON with selected_answer and reasoning. "
            "selected_answer must exactly match one of the supplied options."
        ),
        model=os.getenv("LLM_VALIDATOR_MODEL"),
        temperature=0.0,
        prompt_version=f"necessity_v1_{condition}",
    )

    selected = result.get("selected_answer")
    reasoning = result.get("reasoning", "")

    if not isinstance(selected, str) or not isinstance(reasoning, str):
        raise ValueError("Learner model returned an invalid answer structure.")

    option_lookup = {normalize(option): option for option in question["options"]}
    if normalize(selected) not in option_lookup:
        raise ValueError(f"Model selected an option not in the question: {selected}")

    selected = option_lookup[normalize(selected)]

    return {
        "selected_answer": selected,
        "reasoning": reasoning,
        "correct": normalize(selected) == normalize(question["answer"]),
    }


def run_necessity_tests(relationships_data, concepts_data, quiz_data):
    concepts = {
        item["id"]: item
        for item in concepts_data.get("concepts", [])
    }
    questions_by_concept = {}

    for question in quiz_data.get("items", []):
        questions_by_concept.setdefault(question["concept_id"], []).append(question)

    results = []
    all_without_responses = 0
    all_leakage_responses = 0

    for edge in relationships_data.get("relationships", []):
        source = edge["source"]
        target = edge["target"]

        if source not in concepts or target not in concepts:
            continue

        questions = questions_by_concept.get(target, [])
        if not questions:
            results.append({
                "relationship_id": edge["id"],
                "source": source,
                "target": target,
                "status": "NO_TARGET_QUESTIONS",
                "necessity_gap": None,
                "leakage_rate": None,
                "questions_evaluated": 0,
            })
            continue

        terms = concept_terms(source, concepts[source])
        without_results = []
        with_results = []

        for question in questions:
            without_answer = answer_question(
                question, target, source, "without_prerequisite", concepts
            )
            with_answer = answer_question(
                question, target, source, "with_prerequisite", concepts
            )

            leaked = (
                contains_term(without_answer["reasoning"], terms)
                or contains_term(without_answer["selected_answer"], terms)
            )

            without_results.append({
                "question_id": question["id"],
                **without_answer,
                "leakage_detected": leaked,
            })
            with_results.append({
                "question_id": question["id"],
                **with_answer,
            })

            all_without_responses += 1
            all_leakage_responses += int(leaked)

        count = len(questions)
        score_without = sum(
            item["correct"] for item in without_results
        ) / count
        score_with = sum(
            item["correct"] for item in with_results
        ) / count
        gap = score_with - score_without
        leakage_count = sum(
            item["leakage_detected"] for item in without_results
        )

        results.append({
            "relationship_id": edge["id"],
            "source": source,
            "target": target,
            "status": "MEASURED",
            "questions_evaluated": count,
            "score_without_prerequisite": round(score_without, 4),
            "score_with_prerequisite": round(score_with, 4),
            "necessity_gap": round(gap, 4),
            "leakage_count": leakage_count,
            "leakage_rate": round(leakage_count / count, 4),
            "without_prerequisite_responses": without_results,
            "with_prerequisite_responses": with_results,
        })

    return {
        "schema_version": "1.0",
        "experiment": "simulated_learner_necessity_and_leakage",
        "scoring": {
            "necessity_gap": "score_with_prerequisite - score_without_prerequisite",
            "leakage_rate": "responses_without_prerequisite_with_detected_source_terms / total_without_prerequisite_responses",
        },
        "relationships_evaluated": len(results),
        "overall_leakage_rate": (
            round(all_leakage_responses / all_without_responses, 4)
            if all_without_responses else None
        ),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure prerequisite necessity using a simulated learner."
    )
    parser.add_argument("--relationships", type=Path, required=True)
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--quiz", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cdlp/validation/results/necessity.json"),
    )
    args = parser.parse_args()

    relationships = read_json(args.relationships)
    concepts = read_json(args.concepts)
    quiz = read_json(args.quiz)

    result = run_necessity_tests(relationships, concepts, quiz)
    write_json(args.out, result)

    print(f"Relationships evaluated: {result['relationships_evaluated']}")
    print(f"Overall leakage rate: {result['overall_leakage_rate']}")
    print(f"Results saved to: {args.out}")


if __name__ == "__main__":
    main()
