"""Builder for workbench-mined hard-negative retrieval annotations."""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.hard_negative_retrieval_datasets import HardNegativeRetrievalDataset
from lavis.datasets.datasets.retrieval_datasets import RetrievalEvalDataset


@registry.register_builder("hard_negative_retrieval")
class HardNegativeRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = HardNegativeRetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/hard_negative_retrieval/defaults.yaml"
    }
