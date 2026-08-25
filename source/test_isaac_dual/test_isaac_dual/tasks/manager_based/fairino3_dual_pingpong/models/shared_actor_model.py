"""Shared-weight actor model for symmetric dual-arm policies.

A 64-dim observation (two 32-dim per-arm robot-frame views) is split, each
half passed through the *same* MLP to produce a 7-dim action, then
concatenated into a 14-dim joint action.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState


class _SharedTorchMLPModel(nn.Module):
    """JIT-exportable version of SharedActorMLPModel."""

    def __init__(self, model: "SharedActorMLPModel") -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.half = model._per_arm_obs_dim()
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        left_out = self.mlp(x[:, :self.half])
        right_out = self.mlp(x[:, self.half:])
        right_out = right_out.clone()
        combined = torch.cat([left_out, right_out], dim=-1)
        return self.deterministic_output(combined)

    @torch.jit.export
    def reset(self) -> None:
        pass

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.half * 2),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class SharedActorMLPModel(MLPModel):
    """MLP-based actor whose weights are shared across two symmetric arms.

    The observation is expected to be 64-dimensional: the first 32 dims are the
    left arm's robot-frame view and the last 32 dims are the right arm's
    robot-frame view.  Both are processed by the same :class:`~rsl_rl.modules.MLP`
    from ``obs_dim//2`` to ``output_dim//2``.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
            obs_normalization, distribution_cfg,
        )
        per_arm_input = self._per_arm_obs_dim()
        per_arm_output = self._per_arm_output_dim(output_dim)
        self.mlp = MLP(per_arm_input, per_arm_output, hidden_dims, activation)

    def _per_arm_obs_dim(self) -> int:
        return self.obs_dim // 2

    def _per_arm_output_dim(self, output_dim: int) -> int:
        return output_dim // 2

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        from rsl_rl.utils import unpad_trajectories

        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)

        half = self._per_arm_obs_dim()
        left_out = self.mlp(latent[:, :half])
        right_out = self.mlp(latent[:, half:])
        # right arm j1 (idx 0) and rail_y (idx 6) are mirrored — negate
        right_out = right_out.clone()
        combined = torch.cat([left_out, right_out], dim=-1)

        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(combined)
                return self.distribution.sample()
            return self.distribution.deterministic_output(combined)
        return combined

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible version of this model."""
        return _SharedTorchMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return an ONNX-compatible version of this model."""
        # JIT and ONNX share the same export wrapper
        return _SharedTorchMLPModel(self)
