"""Tests for ROUGE-L metric implementation."""

from __future__ import annotations

import pytest

from rageval.metrics.rouge import _lcs_length, rouge_l_precision, rouge_l_recall, rouge_l_score


class TestLCSLength:
    """Test the Longest Common Subsequence computation."""

    def test_identical_sequences(self) -> None:
        assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_no_overlap(self) -> None:
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_partial_overlap(self) -> None:
        assert _lcs_length(["a", "b", "c", "d"], ["a", "c", "e"]) == 2

    def test_subsequence_not_substring(self) -> None:
        """LCS finds non-contiguous matches."""
        assert _lcs_length(["the", "cat", "sat", "on", "mat"], ["the", "on", "mat"]) == 3

    def test_empty_first(self) -> None:
        assert _lcs_length([], ["a", "b"]) == 0

    def test_empty_second(self) -> None:
        assert _lcs_length(["a", "b"], []) == 0

    def test_both_empty(self) -> None:
        assert _lcs_length([], []) == 0

    def test_single_match(self) -> None:
        assert _lcs_length(["hello"], ["hello"]) == 1

    def test_single_no_match(self) -> None:
        assert _lcs_length(["hello"], ["world"]) == 0


class TestRougeL:
    """Test ROUGE-L F1 score."""

    def test_identical_text(self) -> None:
        score = rouge_l_score("the cat sat on the mat", "the cat sat on the mat")
        assert score == 1.0

    def test_completely_different(self) -> None:
        score = rouge_l_score("hello world", "goodbye universe")
        assert score == 0.0

    def test_partial_overlap_preserves_order(self) -> None:
        """ROUGE-L should reward matching word order."""
        score = rouge_l_score(
            "RAG combines retrieval with generation for better answers",
            "RAG combines retrieval with generation",
        )
        assert score > 0.7

    def test_reordered_text_scores_lower(self) -> None:
        """Reordered words should score lower than ordered matches."""
        ordered = rouge_l_score("the cat sat on the mat", "the cat sat on the mat")
        reordered = rouge_l_score("mat the on sat cat the", "the cat sat on the mat")
        assert ordered > reordered

    def test_empty_candidate(self) -> None:
        assert rouge_l_score("", "some reference") == 0.0

    def test_empty_reference(self) -> None:
        assert rouge_l_score("some candidate", "") == 0.0

    def test_both_empty(self) -> None:
        assert rouge_l_score("", "") == 0.0

    def test_case_insensitive(self) -> None:
        score = rouge_l_score("The Cat SAT", "the cat sat")
        assert score == 1.0

    def test_known_value(self) -> None:
        """Verify against a hand-computed ROUGE-L score.

        Candidate: "the cat on the mat"  (5 tokens)
        Reference: "the cat sat on the mat"  (6 tokens)
        LCS: "the cat on the mat" = 5
        P = 5/5 = 1.0, R = 5/6 = 0.833
        F1 = 2 * (1.0 * 0.833) / (1.0 + 0.833) = 0.909
        """
        score = rouge_l_score("the cat on the mat", "the cat sat on the mat")
        assert score == pytest.approx(0.909, abs=0.01)


class TestRougeLPrecisionRecall:
    """Test precision and recall separately."""

    def test_precision_perfect(self) -> None:
        """Every candidate word is in the reference subsequence."""
        p = rouge_l_precision("the cat", "the cat sat on the mat")
        assert p == 1.0

    def test_recall_perfect(self) -> None:
        """Every reference word is in the candidate subsequence."""
        r = rouge_l_recall("the cat sat on the mat and more", "the cat sat on the mat")
        assert r == 1.0

    def test_precision_empty(self) -> None:
        assert rouge_l_precision("", "reference") == 0.0

    def test_recall_empty(self) -> None:
        assert rouge_l_recall("candidate", "") == 0.0
