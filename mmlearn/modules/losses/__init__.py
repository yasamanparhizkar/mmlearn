"""Loss functions."""

from mmlearn.modules.losses.contrastive import CLIPLoss
from mmlearn.modules.losses.data2vec import Data2VecLoss
from mmlearn.modules.losses.modality import CLIPLossWithModalityLoss


__all__ = ["CLIPLoss", "Data2VecLoss", "CLIPLossWithModalityLoss"]
