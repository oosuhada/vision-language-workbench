"""Retrieval dataset carrying externally mined hard-negative captions."""

from __future__ import annotations

import os
import random

from PIL import Image

from lavis.datasets.datasets.base_dataset import BaseDataset


class HardNegativeRetrievalDataset(BaseDataset):
    """LAVIS-compatible retrieval samples with one mined negative per access."""

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.img_ids: dict[str, int] = {}
        for annotation in self.annotation:
            image_id = str(annotation["image_id"])
            if image_id not in self.img_ids:
                self.img_ids[image_id] = len(self.img_ids)

    def __getitem__(self, index):
        annotation = self.annotation[index]
        negatives = annotation.get("hard_negatives") or []
        if not negatives:
            raise ValueError(f"Hard-negative annotation {index} has no hard_negatives.")
        negative = random.choice(negatives)
        image_path = os.path.join(self.vis_root, annotation["image"])
        image = self.vis_processor(Image.open(image_path).convert("RGB"))
        return {
            "image": image,
            "text_input": self.text_processor(annotation["caption"]),
            "hard_negative_text": self.text_processor(negative["caption"]),
            "hard_negative_similarity": float(negative.get("cosine_similarity", 0.0)),
            "image_id": self.img_ids[str(annotation["image_id"])],
            "instance_id": annotation["instance_id"],
        }
