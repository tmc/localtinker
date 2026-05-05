#!/usr/bin/env python3
"""Minimal Tinker SDK training script for localtinker tests.

Run with the local runner override, for example:

    TINKER_BASE_URL=http://127.0.0.1:8080 python testdata/minimal_tinker_training.py
"""

from __future__ import annotations

import json
import os

import tinker
from tinker import ServiceClient


def main() -> None:
    os.environ.setdefault("TINKER_API_KEY", "tml-local-test")
    os.environ.setdefault("TINKER_BASE_URL", "http://127.0.0.1:8080")

    client = ServiceClient()
    try:
        training = client.create_lora_training_client(
            base_model="Qwen/Qwen3-8B",
            rank=8,
        )
        info = training.get_info()

        data = [
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints([1, 1, 1, 1]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=[1, 1, 1, 1],
                        dtype="int64",
                    ),
                    "weights": tinker.TensorData(
                        data=[1.0, 1.0, 1.0, 1.0],
                        dtype="float32",
                    ),
                },
            )
        ]

        before = loss(training, data)
        training.forward_backward(data, "cross_entropy").result(timeout=120)
        step = training.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(timeout=120)
        after = loss(training, data)

        print(
            json.dumps(
                {
                    "model_id": info.model_id,
                    "loss_before": before,
                    "loss_after": after,
                    "optimizer_metrics": dict(step.metrics),
                },
                sort_keys=True,
            )
        )
    finally:
        client.holder.close()


def loss(training, data) -> float:
    result = training.forward(data, "cross_entropy").result(timeout=120)
    metrics = dict(result.metrics)
    value = metrics.get("loss:mean")
    if value is None:
        raise RuntimeError(f"missing loss:mean metric: {metrics}")
    return value


if __name__ == "__main__":
    main()
