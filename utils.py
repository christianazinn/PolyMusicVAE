"""Shared utilities for PolyMusicVAE."""

from symusic import Score


def extract_notes(score: Score, mode: str = "op") -> set:
    """
    Extract notes from a Score as a set of tuples for F1 comparison.

    Args:
        score: symusic Score object
        mode: "op" for (onset, pitch) or "opd" for (onset, pitch, duration)

    Returns:
        Set of tuples representing notes
    """
    notes = set()
    for track in score.tracks:
        for note in track.notes:
            if mode == "op":
                notes.add((note.time, note.pitch))
            elif mode == "opd":
                notes.add((note.time, note.pitch, note.duration))
            else:
                raise ValueError(f"mode must be 'op' or 'opd', got {mode}")
    return notes


def compute_f1(original_notes: set, reconstructed_notes: set) -> dict:
    """
    Compute precision, recall, and F1 score between two sets of notes.

    Returns:
        Dict with 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'
    """
    tp = len(original_notes & reconstructed_notes)
    fp = len(reconstructed_notes - original_notes)
    fn = len(original_notes - reconstructed_notes)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_batch_f1(
    original_tokens: list[list[int]],
    reconstructed_tokens: list[list[int]],
    tokenizer,
    mode: str = "op",
    pad_id: int = 0,
    bos_id: int = 1,
    eos_id: int = 2,
) -> dict:
    """
    Compute F1 scores for a batch of token sequences.

    Args:
        original_tokens: List of original token sequences
        reconstructed_tokens: List of reconstructed token sequences
        tokenizer: REMI tokenizer for decoding
        mode: "op" or "opd"
        pad_id, bos_id, eos_id: Special token IDs to filter out

    Returns:
        Dict with micro/macro averaged metrics and per-sample scores
    """
    special_tokens = {pad_id, bos_id, eos_id}
    all_scores = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for orig_toks, recon_toks in zip(original_tokens, reconstructed_tokens):
        try:
            # Filter out special tokens
            orig_filtered = [t for t in orig_toks if t not in special_tokens]
            recon_filtered = [t for t in recon_toks if t not in special_tokens]

            if len(orig_filtered) == 0:
                continue

            # Decode to scores
            orig_score = tokenizer.decode([orig_filtered])
            recon_score = tokenizer.decode([recon_filtered])

            # Extract notes and compute F1
            orig_notes = extract_notes(orig_score, mode)
            recon_notes = extract_notes(recon_score, mode)
            scores = compute_f1(orig_notes, recon_notes)

            all_scores.append(scores)
            total_tp += scores["tp"]
            total_fp += scores["fp"]
            total_fn += scores["fn"]

        except Exception:
            # Skip failed decodes
            continue

    if not all_scores:
        return {
            "micro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "macro_f1": 0.0,
            "num_samples": 0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
        }

    # Micro-averaged (note-level)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
               if (micro_precision + micro_recall) > 0 else 0.0

    # Macro-averaged (phrase-level)
    macro_f1 = sum(s["f1"] for s in all_scores) / len(all_scores)

    return {
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "macro_f1": macro_f1,
        "num_samples": len(all_scores),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }
