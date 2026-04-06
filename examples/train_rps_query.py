"""Simple query finetuning example for rock/paper/scissors classification.

Dataset: moondream/classification (subset real-rps)

Requires:
    pip install datasets pillow

Set MOONDREAM_API_KEY and HF_TOKEN environment variables.
"""

import os
import sys
import time
from itertools import cycle
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import moondream as md

QUESTION = "Is this rock, paper, or scissors? Respond with rock, paper, or scissors only."

STEPS = 20
NUM_ROLLOUTS = 4
EVAL_EVERY = 5
LR = 0.001
RANK = 8


def load_examples(target_split):
    return load_dataset(
        "moondream/classification", "real-rps", split="train"
    ).filter(lambda row: row["split"] == target_split)


def evaluate(ft, examples):
    correct, total = 0, 0
    for ex in examples:
        answer = ft.rollouts(
            "query", image=ex["image"], question=QUESTION,
            settings={"temperature": 0.0, "max_tokens": 4},
        )["rollouts"][0]["output"]["answer"]
        correct += answer.strip().lower() == ex["class"]
        total += 1
    return correct / total


def main():
    train_examples = load_examples("train")
    eval_examples = load_examples("valid")

    ft = md.ft(
        api_key=os.environ["MOONDREAM_API_KEY"],
        name=f"rps-query-{int(time.time())}",
        rank=RANK,
    )
    print(f"Created finetune: {ft.finetune_id} ({ft.name})", flush=True)

    requests = (
        (example, {
            "skill": "query",
            "image": example["image"],
            "question": QUESTION,
            "num_rollouts": NUM_ROLLOUTS,
            "settings": {"temperature": 1.0, "max_tokens": 4},
        })
        for example in cycle(train_examples)
    )

    for _, (example, response) in zip(range(STEPS), ft.rollout_stream(requests)):
        rewards = [
            float(r["output"]["answer"].strip().lower() == example["class"])
            for r in response["rollouts"]
        ]
        step = ft.train_step([{
            "mode": "rl",
            "request": response["request"],
            "rollouts": response["rollouts"],
            "rewards": rewards,
        }], lr=LR)
        reward_mean = sum(rewards) / len(rewards)
        print(
            f"step={step['step']} label={example['class']} reward_mean={reward_mean:.3f}",
            flush=True,
        )

        if step["step"] % EVAL_EVERY == 0 or step["step"] == STEPS:
            eval_accuracy = evaluate(ft, eval_examples)
            metrics = ft.log_metrics(
                step=step["step"],
                metrics={"eval/accuracy": eval_accuracy},
            )
            print(
                f"eval step={metrics['step']} accuracy={eval_accuracy:.3f} logged={metrics['logged_count']}",
                flush=True,
            )

    save_result = ft.save_checkpoint()
    checkpoint = save_result["checkpoint"]
    model_id = ft.model(checkpoint["step"])

    print(f"Saved checkpoint: {checkpoint['checkpoint_id']}", flush=True)
    print(f"Model ID: {model_id}", flush=True)


if __name__ == "__main__":
    main()
