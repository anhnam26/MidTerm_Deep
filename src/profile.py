"""Explicit Conv/Linear MACs and counted scalar operations for FruitLite-L.

Inference, batch 1. FLOPs = 2*MACs + elementwise/pooling/BN/activation ops.
Sigmoid is charged 4 scalar operations (negation, exp, addition, division).
Concatenation/reshape have no arithmetic FLOPs. Eval dropout/identity cost zero.
This is an arithmetic convention, NOT hardware instruction count or latency.
"""

import argparse
import collections

import torch
from torch import nn

from .common import ROOT, write_json
from .model import ChannelGate, FruitBlock, FruitLite


def profile(model, size):
    counters = collections.Counter()
    handles = []
    supported = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.Sigmoid, nn.AdaptiveAvgPool2d, nn.Identity, nn.Dropout)
    for module in model.modules():
        if not list(module.children()) and not isinstance(module, supported):
            raise TypeError(f"Uncounted leaf operator: {type(module).__name__}")

    def hook(module, inputs, output):
        elements = output.numel()
        if isinstance(module, nn.Conv2d):
            counters["macs"] += elements * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            if module.bias is not None:
                counters["other_ops"] += elements
        elif isinstance(module, nn.Linear):
            counters["macs"] += elements * module.in_features
            if module.bias is not None:
                counters["other_ops"] += elements
        elif isinstance(module, nn.BatchNorm2d):
            counters["other_ops"] += 4 * elements  # subtract, scale, affine multiply/add
        elif isinstance(module, nn.ReLU):
            counters["other_ops"] += elements
        elif isinstance(module, nn.Sigmoid):
            counters["other_ops"] += 4 * elements
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            counters["other_ops"] += inputs[0].numel()  # sum + division per pool
        elif isinstance(module, (ChannelGate, FruitBlock)):
            counters["other_ops"] += elements  # gate multiplication / residual addition

    for module in model.modules():
        handles.append(module.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(torch.zeros(1, 3, size, size, device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    flops = 2 * counters["macs"] + counters["other_ops"]
    return {"size": size, "output_shape": list(output.shape), "learnable_parameters": parameters,
            "macs": counters["macs"], "other_scalar_ops": counters["other_ops"], "flops": flops,
            "eligible": parameters <= 6_000_000 and flops < 1_000_000_000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    model = FruitLite().to("cuda" if args.cuda else "cpu")
    results = {f"Mid{size}": profile(model, size) for size in (224, 32)}
    write_json(ROOT / "outputs" / "profile.json", results)
    print(results)
    if not all(result["eligible"] for result in results.values()):
        raise SystemExit("Model does not meet the midterm limits")


if __name__ == "__main__":
    main()