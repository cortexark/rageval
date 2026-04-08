"""ROUGE-L metric implementation (pure Python, zero dependencies).

Computes ROUGE-L (Longest Common Subsequence) score between two texts.
This is a standard NLP metric for measuring text similarity that captures
word ordering — unlike Jaccard which treats text as unordered bags of words.

Reference: Lin, C.Y. (2004). ROUGE: A Package for Automatic Evaluation
of Summaries. ACL Workshop on Text Summarization.
"""

from __future__ import annotations


def _lcs_length(seq_a: list[str], seq_b: list[str]) -> int:
    """Compute length of the Longest Common Subsequence.

    Uses O(min(m,n)) space optimization instead of full m*n table.

    Args:
        seq_a: First token sequence.
        seq_b: Second token sequence.

    Returns:
        Length of the LCS.
    """
    if not seq_a or not seq_b:
        return 0

    # Ensure seq_b is the shorter sequence for space optimization
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    m = len(seq_b)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for token_a in seq_a:
        for j, token_b in enumerate(seq_b):
            if token_a == token_b:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(curr[j], prev[j + 1])
        prev, curr = curr, [0] * (m + 1)

    return prev[m]


def rouge_l_score(candidate: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between candidate and reference text.

    ROUGE-L uses the Longest Common Subsequence (LCS) to measure
    similarity while preserving word order information.

    Args:
        candidate: The generated text to evaluate.
        reference: The ground truth reference text.

    Returns:
        ROUGE-L F1 score in [0.0, 1.0].

    Example::

        >>> rouge_l_score("the cat sat on the mat", "the cat on the mat")
        0.9...
    """
    if not candidate or not reference:
        return 0.0

    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    if not cand_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(cand_tokens, ref_tokens)

    if lcs_len == 0:
        return 0.0

    precision = lcs_len / len(cand_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def rouge_l_precision(candidate: str, reference: str) -> float:
    """Compute ROUGE-L precision (LCS / candidate length).

    Args:
        candidate: The generated text.
        reference: The reference text.

    Returns:
        ROUGE-L precision in [0.0, 1.0].
    """
    if not candidate or not reference:
        return 0.0

    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    if not cand_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(cand_tokens, ref_tokens)
    return lcs_len / len(cand_tokens)


def rouge_l_recall(candidate: str, reference: str) -> float:
    """Compute ROUGE-L recall (LCS / reference length).

    Args:
        candidate: The generated text.
        reference: The reference text.

    Returns:
        ROUGE-L recall in [0.0, 1.0].
    """
    if not candidate or not reference:
        return 0.0

    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    if not cand_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(cand_tokens, ref_tokens)
    return lcs_len / len(ref_tokens)
