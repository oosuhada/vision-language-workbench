"""BLIP retrieval with an additional externally mined hard-negative objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from lavis.common.registry import registry
from lavis.models.blip_models.blip_outputs import BlipOutput
from lavis.models.blip_models.blip_retrieval import BlipRetrieval
from workbench.hard_negative_training import hard_negative_margin_loss


@dataclass
class BlipHardNegativeOutput(BlipOutput):
    loss_hard_negative: Optional[torch.FloatTensor] = None


@registry.register_model("blip_retrieval_hard_negative")
class BlipRetrievalHardNegative(BlipRetrieval):
    """Reuse BLIP ITC/ITM and add a mined-negative cosine margin term."""

    PRETRAINED_MODEL_CONFIG_DICT = {
        "coco": "configs/models/blip_retrieval_hard_negative_coco.yaml",
        "flickr": "configs/models/blip_retrieval_hard_negative_coco.yaml",
    }

    def __init__(self, *args, hard_negative_margin=0.2, hard_negative_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.hard_negative_margin = float(hard_negative_margin)
        self.hard_negative_weight = float(hard_negative_weight)

    @classmethod
    def from_config(cls, cfg=None):
        model = super().from_config(cfg)
        model.hard_negative_margin = float(cfg.get("hard_negative_margin", 0.2))
        model.hard_negative_weight = float(cfg.get("hard_negative_weight", 1.0))
        return model

    def forward(self, samples):
        base_output = super().forward(samples)
        negative_captions = samples.get("hard_negative_text")
        if negative_captions is None:
            return base_output

        intermediate = base_output.intermediate_output
        image_features = F.normalize(self.vision_proj(intermediate.image_embeds[:, 0, :]), dim=-1)
        positive_features = F.normalize(self.text_proj(intermediate.text_embeds[:, 0, :]), dim=-1)

        negative_tokens = self.tokenizer(
            negative_captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image_features.device)
        negative_output = self.text_encoder.forward_text(negative_tokens)
        negative_features = F.normalize(self.text_proj(negative_output.last_hidden_state[:, 0, :]), dim=-1)

        hardness = samples.get("hard_negative_similarity")
        if hardness is not None:
            hardness = torch.as_tensor(hardness, device=image_features.device, dtype=image_features.dtype)
            hardness = hardness.clamp_min(0)

        loss_hard_negative = hard_negative_margin_loss(
            image_features,
            positive_features,
            negative_features,
            margin=self.hard_negative_margin,
            hardness=hardness,
        )
        total_loss = base_output.loss + self.hard_negative_weight * loss_hard_negative
        return BlipHardNegativeOutput(
            loss=total_loss,
            loss_itc=base_output.loss_itc,
            loss_itm=base_output.loss_itm,
            loss_lm=base_output.loss_lm,
            sims=base_output.sims,
            intermediate_output=base_output.intermediate_output,
            loss_hard_negative=loss_hard_negative,
        )
