"""
Convert the M3ED annotation file into the per-utterance vote table used by the experiments.

Input:  DATA_ROOT/raw/m3ed/annotation.json
        annotation[show][episode]['Dialog'][utt_id]['EmoAnnotation'] = {'EmoAnnotator1', 'EmoAnnotator2',
        'EmoAnnotator3', ...}
Output: DATA_ROOT/processed/m3ed_softlabels.parquet with one row per utterance:
        dataset_source, dialog_id, turn_id, speaker_id, utterance, hard_label,
        p_dist (vote shares in the order neutral, joy, sadness, fear, anger, surprise, disgust), n_raters
Utterances are ordered by their utterance id within each episode; utterances without a valid vote are dropped.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

EMOTIONS = ["neutral", "joy", "sadness", "fear", "anger", "surprise", "disgust"]
K = len(EMOTIONS)
M3ED_TO_EKMAN = {
    "Neutral": "neutral",
    "Happy": "joy",
    "Sad": "sadness",
    "Fear": "fear",
    "Anger": "anger",
    "Surprise": "surprise",
    "Disgust": "disgust",
}
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}


def parse_annotation(annot: dict):
    """Vote shares over EMOTIONS and the number of valid annotators."""
    votes = np.zeros(K)
    n = 0
    for key in ("EmoAnnotator1", "EmoAnnotator2", "EmoAnnotator3"):
        e = M3ED_TO_EKMAN.get(annot.get(key))
        if e is None:
            continue
        votes[EMO2IDX[e]] += 1
        n += 1
    if n == 0:
        return votes, 0
    return votes / n, n


def build_m3ed_dataframe(path: Path) -> pd.DataFrame:
    data = json.load(open(path, encoding="utf-8"))
    rows = []
    for show_name, show in data.items():
        for ep_id, ep in show.items():
            dialog = ep.get("Dialog", {})
            for turn_id, utt in enumerate(sorted(dialog)):
                u = dialog[utt]
                p, n_raters = parse_annotation(u.get("EmoAnnotation", {}))
                if n_raters == 0:
                    continue
                rows.append({
                    "dataset_source": "m3ed",
                    "dialog_id": f"{show_name}::{ep_id}",
                    "turn_id": turn_id,
                    "speaker_id": u.get("Speaker", "A"),
                    "utterance": u.get("Text", ""),
                    "hard_label": EMOTIONS[int(p.argmax())],
                    "p_dist": p.tolist(),
                    "n_raters": n_raters,
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_m3ed_dataframe(config.M3ED_ANNOTATION)
    config.M3ED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.M3ED_PARQUET)
    print(f"{len(df)} utterances, {df['dialog_id'].nunique()} dialogues -> {config.M3ED_PARQUET}")
