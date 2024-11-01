"""Modules for pretraining, downstream and evaluation tasks."""

from mmlearn.tasks.contrastive_pretraining import ContrastivePretraining
from mmlearn.tasks.zero_shot_classification import ZeroShotClassification
from mmlearn.tasks.zero_shot_retrieval import ZeroShotCrossModalRetrieval
from mmlearn.tasks.linear_probing import LinearClassifierModule
from mmlearn.tasks.contrastive_pretraining_modality import ContrastivePretrainingModality


__all__ = [
    "ContrastivePretraining",
    "LinearClassifierModule",
    "ZeroShotCrossModalRetrieval",
    "ZeroShotClassification",
    "ContrastivePretrainingModality",
]
