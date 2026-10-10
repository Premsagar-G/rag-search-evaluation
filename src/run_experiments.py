"""
Task 4: Experiment grid + Task 6: Logging.

Runs >=6 configurations that vary one factor at a time (A/B style) over the
baseline, executes the full Task-5 question set (with memory reset between
groups) for each config, and logs every answer to CSV + JSONL with run
metadata (chunk_size, overlap, embedding, prompt_id) for later analysis.
"""
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import RagPipeline
from questions import GROUPS

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# BASELINE: chunk_size=600, overlap=15%, embedding=open_source, prompt=P1
BASELINE = dict(chunk_size=600, overlap_pct=0.15, embedding="open_source", prompt="P1")

# One-factor-at-a-time grid (>=6 runs), per Task 4 instructions
RUNS = [
    dict(run_id="R1_baseline",            chunk_size=600,  overlap_pct=0.15, embedding="open_source", prompt="P1"),
    dict(run_id="R2_chunk_300",           chunk_size=300,  overlap_pct=0.15, embedding="open_source", prompt="P1"),
    dict(run_id="R3_chunk_1000",          chunk_size=1000, overlap_pct=0.15, embedding="open_source", prompt="P1"),
    dict(run_id="R4_overlap_20pct",       chunk_size=600,  overlap_pct=0.20, embedding="open_source", prompt="P1"),
    dict(run_id="R5_overlap_10pct",       chunk_size=600,  overlap_pct=0.10, embedding="open_source", prompt="P1"),
    dict(run_id="R6_embedding_proprietary", chunk_size=600, overlap_pct=0.15, embedding="proprietary", prompt="P1"),
    dict(run_id="R7_prompt_P2",           chunk_size=600,  overlap_pct=0.15, embedding="open_source", prompt="P2"),
    dict(run_id="R8_chunk300_proprietary", chunk_size=300, overlap_pct=0.20, embedding="proprietary", prompt="P2"),
]


def run_all():
    csv_path = os.path.join(LOG_DIR, "run_log.csv")
    jsonl_path = os.path.join(LOG_DIR, "run_log.jsonl")

    fieldnames = [
        "run_id", "chunk_size", "overlap_pct", "embedding", "prompt",
        "group", "turn_index", "question_type", "question", "answer",
        "citations", "used_chunks", "tokens", "latency_sec", "retrieval_top_score",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as cf, \
         open(jsonl_path, "w", encoding="utf-8") as jf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        for run_cfg in RUNS:
            run_id = run_cfg["run_id"]
            pipe = RagPipeline(
                chunk_size=run_cfg["chunk_size"],
                overlap_pct=run_cfg["overlap_pct"],
                embedding_name=run_cfg["embedding"],
                k=4,
            )
            print(f"== {run_id} | chunks={len(pipe.chunks)} | backend={pipe.store.backend} ==")

            for group in GROUPS:
                pipe.memory.reset()  # Task 5: reset memory between groups
                for i, turn in enumerate(group["turns"]):
                    result = pipe.ask(turn["q"], run_cfg["prompt"])
                    row = {
                        "run_id": run_id,
                        "chunk_size": run_cfg["chunk_size"],
                        "overlap_pct": run_cfg["overlap_pct"],
                        "embedding": run_cfg["embedding"],
                        "prompt": run_cfg["prompt"],
                        "group": group["label"],
                        "turn_index": i + 1,
                        "question_type": turn["type"],
                        "question": turn["q"],
                        "answer": result["formatted_answer"].replace("\n", " | "),
                        "citations": ";".join(result["citations"]),
                        "used_chunks": ";".join(result["used_chunks"]),
                        "tokens": result["tokens"],
                        "latency_sec": result["latency_sec"],
                        "retrieval_top_score": round(result["retrieval_top_score"], 4),
                    }
                    writer.writerow(row)
                    jf.write(json.dumps({**row, "config": result["config"]}) + "\n")

    print(f"\nWrote logs to:\n  {csv_path}\n  {jsonl_path}")
    return csv_path, jsonl_path


if __name__ == "__main__":
    run_all()
