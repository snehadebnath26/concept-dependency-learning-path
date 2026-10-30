
from cdlp.discovery.llm_discovery import (
    _mentions_concept,
    _normalise,
    _supports_direction
)


def test_normalise_removes_punctuation():
    assert _normalise("Binary Search Tree!") == "binary search tree"


def test_mentions_concept_matches_full_term():
    assert _mentions_concept(
        "An AVL tree is a self-balancing binary search tree.",
        ["binary search tree"]
    )


def test_mentions_concept_rejects_partial_word():
    assert not _mentions_concept(
        "A graph generalises a tree.",
        ["trees"]
    )


def test_avl_definition_mentions_both_concepts():
    quote = "An AVL tree is a self-balancing binary search tree."
    assert _mentions_concept(quote, ["avl tree"])
    assert _mentions_concept(quote, ["binary search tree"])


def test_binary_search_tree_requires_binary_tree():
    quote = "A binary search tree is a binary tree."
    assert _supports_direction("binary tree", "binary search tree", quote)


def test_reversed_binary_tree_relationship_is_rejected():
    quote = "A binary search tree is a binary tree."
    assert not _supports_direction("binary search tree", "binary tree", quote)


def test_graph_generalises_tree_supports_trees_to_graphs():
    quote = (
        "A graph generalises a tree: nodes connected by edges, "
        "possibly with cycles and no root."
    )
    assert _supports_direction("trees", "graphs", quote)


def test_graph_generalisation_does_not_support_reverse_direction():
    quote = "A graph generalises a tree: nodes connected by edges."
    assert not _supports_direction("graphs", "trees", quote)
