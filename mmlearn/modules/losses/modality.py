"""Implementations of the contrastive loss and its variants."""

from typing import Dict, Tuple, Optional

import torch
import torch.distributed as dist
from hydra_zen import store
import itertools
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torchmetrics.utilities.compute import _safe_matmul
from torchmetrics.utilities.distributed import gather_all_tensors

from mmlearn.datasets.core import Modalities, find_matching_indices

@dataclass
class LossPairSpec:
    """Specification for a pair of modalities to compute the contrastive loss."""

    modalities: Tuple[str, str]
    weight: float = 1.0

class CLIPLossWithModalityLoss(nn.Module):
    """CLIP Loss module.

    Parameters
    ----------
    l2_normalize : bool, default=False
        Whether to L2 normalize the features.
    local_loss : bool, default=False
        Whether to calculate the loss locally i.e. `local_features@global_features`.
    gather_with_grad : bool, default=False
        Whether to gather tensors with gradients.
    cache_labels : bool, default=False
        Whether to cache the labels.

    """

    def __init__(
        self,
        modality_loss: bool = True,
        l2_normalize: bool = False,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
    ):
        """Initialize the loss."""
        super().__init__()
        self.modality_loss = modality_loss
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.l2_normalize = l2_normalize

        # cache state
        self._prev_num_logits = 0
        self._labels: Dict[torch.device, torch.Tensor] = {}

    def _get_ground_truth(
        self, device: torch.device, num_logits: int, rank: int, world_size: int
    ) -> torch.Tensor:
        """Return the ground-truth labels.

        Parameters
        ----------
        device : torch.device
            Device to store the labels.
        num_logits : int
            Number of logits.
        rank : int
            Rank of the current process.
        world_size : int
            Number of processes.

        Returns
        -------
        torch.Tensor
            Ground-truth labels.
        """
        # calculate ground-truth and cache if enabled
        if self._prev_num_logits != num_logits or device not in self._labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if world_size > 1 and self.local_loss:
                labels = labels + num_logits * rank
            if self.cache_labels:
                self._labels[device] = labels
                self._prev_num_logits = num_logits
        else:
            labels = self._labels[device]
        return labels

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        modality_loss_pairs: Optional[LossPairSpec] = None,
        embedding_pair_indices: Optional[
            list[tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> torch.Tensor:
        available_modalities = list(embeddings.keys())
        if modality_loss_pairs is None:
            modality_loss_pairs = [
                LossPairSpec(modalities=(m1, m2))
                for m1, m2 in itertools.combinations(available_modalities, 2)
            ]

        if self.l2_normalize:
            embeddings = {k: F.normalize(v, p=2, dim=-1) for k, v in embeddings.items()}

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if world_size > 1 else 0

        gathered_embeddings = embeddings.copy()
        if world_size > 1:
            gathered_embeddings = {
                k: gather_features(v, self.local_loss, self.gather_with_grad, rank)
                for k, v in embeddings.items()
            }

        losses = []
        for idx, loss_pairs in enumerate(modality_loss_pairs):
            if world_size > 1:
                modality_a = Modalities.get_modality(loss_pairs.modalities[0])
                modality_b = Modalities.get_modality(loss_pairs.modalities[1])
                if modality_a.name not in gathered_embeddings.keys() or modality_b.name not in gathered_embeddings.keys():
                    continue
                all_features_a = gathered_embeddings[modality_a.name]
                all_features_b = gathered_embeddings[modality_b.name]

            modality_a = Modalities.get_modality(loss_pairs.modalities[0])
            modality_b = Modalities.get_modality(loss_pairs.modalities[1])
            if modality_a.name not in embeddings.keys() or modality_b.name not in embeddings.keys():
                continue
            features_a = embeddings[modality_a.name]
            features_b = embeddings[modality_b.name]

            if embedding_pair_indices is not None:
                indices_a, indices_b = embedding_pair_indices[idx]

                if world_size > 1:
                    all_features_a = all_features_a[indices_a]
                    all_features_b = all_features_b[indices_b]

                features_a = features_a[indices_a]
                features_b = features_b[indices_b]

            # compute logits
            if world_size > 1:
                if self.local_loss:
                    logits_per_feature_a = _safe_matmul(
                        features_a, all_features_b
                    )
                    logits_per_feature_b = _safe_matmul(
                        features_b, all_features_a
                    )
                else:
                    logits_per_feature_a = _safe_matmul(
                        all_features_a, all_features_b
                    )
                    logits_per_feature_b = logits_per_feature_a.T
            else:
                logits_per_feature_a = _safe_matmul(
                    features_a, features_b
                )
                logits_per_feature_b = _safe_matmul(
                    features_b, features_a
                )

            labels = self._get_ground_truth(
                features_b.device,
                logits_per_feature_a.shape[0],
                rank=rank,
                world_size=world_size,
            )

            losses.append(
                (
                    (
                        F.cross_entropy(logits_per_feature_a, labels)
                        + F.cross_entropy(logits_per_feature_b, labels)
                    )
                    / 2
                )
                * loss_pairs.weight
            )
            

        if self.modality_loss:
            if world_size > 1:
                all_features = torch.cat(
                    [gathered_embeddings[k] for k in gathered_embeddings], dim=0
                )
            else:
                all_features = torch.cat([embeddings[k] for k in embeddings], dim=0)

            positive_indices = torch.tensor(
                [
                    (i, j)
                    if idx == 0
                    else (
                        i + gathered_embeddings[available_modalities[idx - 1]].size(0),
                        j + gathered_embeddings[available_modalities[idx - 1]].size(0),
                    )
                    for idx, k in enumerate(gathered_embeddings)
                    for i, j in itertools.combinations(
                        range(gathered_embeddings[k].size(0)), 2
                    )
                ],
                device=all_features.device,
            )
            logits = _safe_matmul(all_features, all_features)
            logits[torch.eye(all_features.size(0)).bool()] = float("inf")

            target = torch.eye(all_features.size(0))
            target[positive_indices[:, 0], positive_indices[:, 1]] = 1
            target = target.to(logits.device)
            modality_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.sigmoid(), target, reduction="none"
            )

            target_pos = target.bool()
            target_neg = ~target_pos

            # loss_pos and loss_neg below contain non-zero values only for those elements
            # that are positive pairs and negative pairs respectively.
            device = logits.device
            modality_loss = modality_loss.to(device)
            target_pos = target_pos.to(device)  
            target_neg = target_neg.to(device)  
            loss_pos = torch.zeros(logits.size(0), logits.size(0), device=device).masked_scatter(
                target_pos, modality_loss[target_pos]
            )
            loss_neg = torch.zeros(logits.size(0), logits.size(0), device=device).masked_scatter(
                target_neg, modality_loss[target_neg]
            )

            loss_pos = loss_pos.sum(dim=1)
            loss_neg = loss_neg.sum(dim=1)
            num_pos = target.sum(dim=1)
            num_neg = logits.size(0) - num_pos

            losses.append(((loss_pos / num_pos) + (loss_neg / num_neg)).mean())

        return torch.stack(losses).sum()


def gather_features(
    features: torch.Tensor,
    local_loss: bool = False,
    gather_with_grad: bool = False,
    rank: int = 0,
) -> torch.Tensor:
    """Gather features across all processes.

    Parameters
    ----------
    features : torch.Tensor
        First feature tensor to gather.
    local_loss : bool, default=False
        Whether to calculate the loss locally i.e.
        `matmul(local_features, global_features)`. If False, this method ensures
        that the gathered features contain local features for the current rank.
    gather_with_grad : bool, default=False
        Whether to gather tensors with gradients.
    rank : int, default=0
        Rank of the current process.

    Returns
    -------
    torch.Tensor
        Gathered features.
    """
    if gather_with_grad:
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
    else:
        gathered_features = gather_all_tensors(features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_features[rank] = features
        all_features = torch.cat(gathered_features, dim=0)

    return all_features