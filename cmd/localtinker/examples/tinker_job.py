#!/usr/bin/env python3
"""Run a small Tinker SDK training job against localtinker."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import tinker
from tinker import ServiceClient

RESULT_TIMEOUT = 120


@dataclass(frozen=True)
class Preset:
    steps: int


PRESETS = {
    "short": Preset(
        steps=4,
    ),
    "long": Preset(
        steps=24,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="short")
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "tml-local-test"))
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=0, help="override preset step count")
    parser.add_argument("--learning-rate", type=float, default=0, help="override preset learning rate")
    args = parser.parse_args()

    os.environ["TINKER_BASE_URL"] = args.base_url
    os.environ["TINKER_API_KEY"] = args.api_key

    preset = PRESETS[args.preset]
    steps = args.steps or preset.steps
    learning_rate = args.learning_rate or 1e-4

    client = ServiceClient()
    try:
        try:
            training = client.create_lora_training_client(
                base_model=args.base_model,
                rank=args.rank,
            )
            info = training.get_info()
            print(
                json.dumps(
                    {
                        "event": "training_created",
                        "model_id": info.model_id,
                        "tokenizer_id": info.model_data.tokenizer_id,
                    },
                    sort_keys=True,
                )
            )
            data = training_data(training)
            validate_cross_entropy_data(data)
            before = loss(training, data, info.model_id, "before")
            step_metrics = []
            for i in range(steps):
                forward_backward_future = training.forward_backward(data, "cross_entropy")
                print(
                    json.dumps(
                        future_event("forward_backward", info.model_id, forward_backward_future, i + 1),
                        sort_keys=True,
                    )
                )
                forward_backward_future.result(timeout=RESULT_TIMEOUT)
                optim_future = training.optim_step(tinker.AdamParams(learning_rate=learning_rate))
                print(
                    json.dumps(
                        future_event("optim_step", info.model_id, optim_future, i + 1),
                        sort_keys=True,
                    )
                )
                result = optim_future.result(timeout=RESULT_TIMEOUT)
                metrics = dict(result.metrics)
                metrics["step"] = i + 1
                step_metrics.append(metrics)
                print(json.dumps({"event": "step", "model_id": info.model_id, "metrics": metrics}, sort_keys=True))
            after = loss(training, data, info.model_id, "after")
            summary = {
                "event": "summary",
                "preset": args.preset,
                "base_url": args.base_url,
                "model_id": info.model_id,
                "tokenizer_id": info.model_data.tokenizer_id,
                "steps": steps,
                "learning_rate": learning_rate,
                "loss_before": before,
                "loss_after": after,
                "loss_delta": after - before,
                "optimizer_backend_mlx_seen": any(m.get("optimizer_backend:mlx") == 1 for m in step_metrics),
            }
            print(json.dumps(summary, sort_keys=True))
            if after >= before:
                raise SystemExit(f"loss did not decrease: before={before} after={after}")
            if not summary["optimizer_backend_mlx_seen"]:
                raise SystemExit("MLX optimizer backend was not reported")
        except Exception as e:
            print(json.dumps({"event": "error", "error_type": type(e).__name__, "error": str(e)}, sort_keys=True))
            raise
    finally:
        client.holder.close()


def loss(training, data, model_id: str, label: str) -> float:
    future = training.forward(data, "cross_entropy")
    print(json.dumps(future_event("forward", model_id, future, label=label), sort_keys=True))
    result = future.result(timeout=RESULT_TIMEOUT)
    metrics = dict(result.metrics)
    loss_value = metrics.get("loss:mean", metrics.get("loss:sum"))
    if loss_value is None:
        raise ValueError(f"forward result did not include a loss metric: {sorted(metrics)}")
    print(
        json.dumps(
            {"event": "loss", "label": label, "model_id": model_id, "loss": loss_value, "metrics": metrics},
            sort_keys=True,
        )
    )
    return loss_value


def future_event(op: str, model_id: str, future, step: int | None = None, label: str = "") -> dict:
    event = {
        "event": "future_submitted",
        "op": op,
        "model_id": model_id,
        "future_wrapper": type(future).__name__,
    }
    if step is not None:
        event["step"] = step
    if label:
        event["label"] = label
    return event


def training_data(training) -> list[tinker.Datum]:
    tokenizer = training.get_tokenizer()
    text = "Tinker smoke test. The answer is blue. Tinker smoke test. The answer is blue."
    tokens = tokenizer.encode(text)[:32]
    if len(tokens) < 2:
        raise ValueError("smoke text produced fewer than 2 tokens")
    targets = tokens[1:]
    return [
        tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=targets,
                    dtype="int64",
                    shape=[len(targets)],
                ),
                "weights": tinker.TensorData(
                    data=[1.0] * len(targets),
                    dtype="float32",
                    shape=[len(targets)],
                ),
            },
        )
    ]


def validate_cross_entropy_data(data: list[tinker.Datum]) -> None:
    for i, datum in enumerate(data):
        keys = set(datum.loss_fn_inputs)
        if keys - {"target_tokens", "weights"}:
            raise ValueError(f"datum {i}: unsupported cross_entropy loss_fn_inputs keys: {sorted(keys)}")
        if "target_tokens" not in datum.loss_fn_inputs:
            raise ValueError(f"datum {i}: missing target_tokens")

        target = datum.loss_fn_inputs["target_tokens"]
        target_len = tensor_len(target)
        if "weights" in datum.loss_fn_inputs:
            weight_len = tensor_len(datum.loss_fn_inputs["weights"])
            if weight_len != target_len:
                raise ValueError(
                    f"datum {i}: weights length {weight_len} does not match target_tokens length {target_len}"
                )

        print(
            json.dumps(
                {
                    "event": "loss_input_preflight",
                    "datum": i,
                    "keys": sorted(keys),
                    "target_shape": target.shape or [target_len],
                    "target_len": target_len,
                    "model_input_len": datum.model_input.length,
                },
                sort_keys=True,
            )
        )


def tensor_len(tensor: tinker.TensorData) -> int:
    if tensor.shape is None:
        return len(tensor.data)
    n = 1
    for dim in tensor.shape:
        n *= dim
    if n != len(tensor.data):
        raise ValueError(f"tensor shape {tensor.shape} has {n} elements but data has {len(tensor.data)}")
    return n


if __name__ == "__main__":
    main()
