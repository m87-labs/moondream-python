"""Simple query finetuning example for rock/paper/scissors classification.

Dataset: moondream/classification (subset real-rps)

Requires:
    pip install datasets pillow

Set `MOONDREAM_API_KEY` and `HF_TOKEN`, or pass them via flags.
"""

import argparse
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
from moondream.finetune import DEFAULT_TUNING_ENDPOINT


DATASET_NAME = "moondream/classification"
DATASET_SUBSET = "real-rps"
QUESTION = "Is this rock, paper, or scissors? Respond with rock, paper, or scissors only."
TRAIN_SETTINGS = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 4}
EVAL_SETTINGS = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4}
VALID_LABELS = ("rock", "paper", "scissors")


def normalize_label(text: str) -> str:
    lowered = " ".join(text.strip().lower().split())
    if "scissor" in lowered:
        return "scissors"
    for label in VALID_LABELS:
        if label in lowered:
            return label
    return ""


def reward(answer: str, expected: str) -> float:
    return 1.0 if normalize_label(answer) == normalize_label(expected) else 0.0


def iter_examples(target_split: str, hf_token: str):
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_SUBSET,
        split="train",
        streaming=True,
        token=hf_token,
    )

    while True:
        for row in dataset:
            row_split = str(row.get("split", "")).lower()
            if row_split != target_split:
                continue

            label = normalize_label(str(row.get("class", "")))
            image = row.get("image")
            if not label or image is None:
                continue

            yield {"image": image, "answer": label}


def evaluate(ft, examples: list[dict]) -> float:
    if not examples:
        return 0.0

    correct = 0
    for example in examples:
        response = ft.rollouts(
            md.types.RolloutRequest(
                skill="query",
                image=example["image"],
                question=QUESTION,
                num_rollouts=1,
                settings=EVAL_SETTINGS,
            )
        )
        answer = response["rollouts"][0]["output"].get("answer", "")
        correct += int(reward(answer, example["answer"]))

    return correct / len(examples)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("MOONDREAM_API_KEY"))
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--endpoint", default=DEFAULT_TUNING_ENDPOINT)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Pass --api-key or set MOONDREAM_API_KEY")
    if not args.hf_token:
        raise SystemExit("Pass --hf-token or set HF_TOKEN")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.num_rollouts < 1 or args.num_rollouts > 16:
        raise SystemExit("--num-rollouts must be between 1 and 16")
    if args.eval_every < 1:
        raise SystemExit("--eval-every must be at least 1")
    if args.eval_samples < 1:
        raise SystemExit("--eval-samples must be at least 1")

    train_examples = iter_examples("train", args.hf_token)
    eval_examples = list(islice(iter_examples("valid", args.hf_token), args.eval_samples))
    if not eval_examples:
        raise SystemExit("Could not load any eval examples")

    finetune_name = f"rps-query-{int(time.time())}"
    ft = md.ft(
        api_key=args.api_key,
        name=finetune_name,
        rank=args.rank,
        endpoint=args.endpoint,
    )

    print(f"Created finetune: {ft.finetune_id} ({ft.name})", flush=True)

    def make_requests(examples, num_rollouts, settings):
        for example in examples:
            yield example, md.types.RolloutRequest(
                skill="query",
                image=example["image"],
                question=QUESTION,
                num_rollouts=num_rollouts,
                settings=settings,
            )

    requests = make_requests(train_examples, args.num_rollouts, TRAIN_SETTINGS)

    for example, response in ft.rollout_stream(islice(requests, args.steps)):
        rewards = [
            reward(rollout["output"].get("answer", ""), example["answer"])
            for rollout in response["rollouts"]
        ]
        step = ft.train_step([{
            "mode": "rl",
            "request": response["request"],
            "rollouts": response["rollouts"],
            "rewards": rewards,
        }], lr=args.lr)
        reward_mean = sum(rewards) / len(rewards)
        print(
            f"step={step['step']} label={example['answer']} reward_mean={reward_mean:.3f}",
            flush=True,
        )

        if step["step"] % args.eval_every == 0 or step["step"] == args.steps:
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
