"""Simple query finetuning example for rock/paper/scissors classification.

Dataset: moondream/classification (subset real-rps)

Requires:
    pip install datasets pillow

Set MOONDREAM_API_KEY and HF_TOKEN environment variables.
"""

import os
import sys
import time
from itertools import islice
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
EVAL_SAMPLES = 16
LR = 0.001
RANK = 8


def reward(answer: str, expected: str) -> float:
    return 1.0 if answer.strip().lower() == expected else 0.0


def iter_examples(target_split: str):
    dataset = load_dataset(
        "moondream/classification",
        "real-rps",
        split="train",
        streaming=True,
    ).filter(lambda row: row.get("split", "").lower() == target_split)

    while True:
        yield from dataset


def evaluate(ft, examples):
    correct = sum(
        ft.rollouts(
            "query", image=ex["image"], question=QUESTION,
            settings={"temperature": 0.0, "max_tokens": 4},
        )["rollouts"][0]["output"].get("answer", "").strip().lower() == ex["class"]
        for ex in examples
    )
    return correct / len(examples)


def main():
    train_examples = iter_examples("train")
    eval_examples = list(islice(iter_examples("valid"), EVAL_SAMPLES))
    assert eval_examples, "Could not load any eval examples"

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
        for example in train_examples
    )

    for example, response in ft.rollout_stream(islice(requests, STEPS)):
        rewards = [
            reward(rollout["output"].get("answer", ""), example["class"])
            for rollout in response["rollouts"]
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
