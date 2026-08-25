"""Research-oriented LoRA target policies for arbitrary LAVIS torch modules.

This module does not replace LAVIS architectures. It only wraps selected
``torch.nn.Linear`` layers with a standard low-rank residual so target
placement and rank become explicit experiment variables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import torch
from torch import nn
import yaml


@dataclass(frozen=True)
class LoRAPolicy:
    name: str
    rank: int
    alpha: float
    dropout: float = 0.0
    target_suffixes: tuple[str, ...] = ()
    target_regex: tuple[str, ...] = ()
    freeze_base: bool = True

    @classmethod
    def load(cls, path: str | Path) -> "LoRAPolicy":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("LoRA policy must contain a mapping/object.")
        rank = int(raw.get("rank", 0))
        alpha = float(raw.get("alpha", rank))
        dropout = float(raw.get("dropout", 0.0))
        suffixes = tuple(str(value) for value in (raw.get("target_suffixes") or []))
        regexes = tuple(str(value) for value in (raw.get("target_regex") or []))
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1.")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be > 0.")
        if not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1).")
        if not suffixes and not regexes:
            raise ValueError("LoRA policy needs target_suffixes and/or target_regex.")
        return cls(
            name=str(raw.get("name") or Path(path).stem),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_suffixes=suffixes,
            target_regex=regexes,
            freeze_base=bool(raw.get("freeze_base", True)),
        )

    def matches(self, module_name: str) -> bool:
        if any(module_name == suffix or module_name.endswith(f".{suffix}") for suffix in self.target_suffixes):
            return True
        return any(re.search(pattern, module_name) is not None for pattern in self.target_regex)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_suffixes"] = list(self.target_suffixes)
        result["target_regex"] = list(self.target_regex)
        return result


class LoRALinear(nn.Module):
    """A low-rank residual around an existing ``nn.Linear`` layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0, freeze_base: bool = True):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.scaling * self.lora_B(self.lora_A(self.dropout(inputs)))

    @property
    def adapter_parameters(self) -> int:
        return self.rank * (self.base.in_features + self.base.out_features)


def discover_lora_targets(model: nn.Module, policy: LoRAPolicy) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and policy.matches(name):
            targets.append(
                {
                    "name": name,
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                    "base_parameters": int(sum(p.numel() for p in module.parameters())),
                    "lora_parameters": int(policy.rank * (module.in_features + module.out_features)),
                }
            )
    return targets


def _parent_and_leaf(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(model: nn.Module, policy: LoRAPolicy) -> dict[str, Any]:
    """Replace matching linear layers in-place and return parameter budgeting."""
    targets = discover_lora_targets(model, policy)
    if not targets:
        raise ValueError(f"LoRA policy '{policy.name}' matched no nn.Linear modules.")
    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    if policy.freeze_base:
        for parameter in model.parameters():
            parameter.requires_grad = False
    for target in targets:
        parent, leaf = _parent_and_leaf(model, str(target["name"]))
        base = getattr(parent, leaf)
        setattr(parent, leaf, LoRALinear(base, policy.rank, policy.alpha, policy.dropout, policy.freeze_base))
    adapter_parameters = int(sum(int(target["lora_parameters"]) for target in targets))
    trainable_parameters = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    return {
        "policy": policy.as_dict(),
        "matched_modules": len(targets),
        "targets": targets,
        "base_parameters": total_parameters,
        "adapter_parameters": adapter_parameters,
        "trainable_parameters_after_injection": trainable_parameters,
        "adapter_fraction_of_base": adapter_parameters / max(total_parameters, 1),
    }


def compare_lora_policies(model: nn.Module, policies: Iterable[LoRAPolicy]) -> list[dict[str, Any]]:
    """Estimate multiple target/rank policies without mutating ``model``."""
    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    rows: list[dict[str, Any]] = []
    for policy in policies:
        targets = discover_lora_targets(model, policy)
        adapter_parameters = int(sum(int(target["lora_parameters"]) for target in targets))
        rows.append(
            {
                "policy": policy.name,
                "rank": policy.rank,
                "alpha": policy.alpha,
                "matched_modules": len(targets),
                "adapter_parameters": adapter_parameters,
                "adapter_fraction_of_base": adapter_parameters / max(total_parameters, 1),
                "target_names": [target["name"] for target in targets],
            }
        )
    return rows
