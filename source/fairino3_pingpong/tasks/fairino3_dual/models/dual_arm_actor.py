"""Dual-arm actor model with separate MLPs per arm, plus pretrained left-arm loading.

Observation (64-dim) is split:
  - left_obs  (32) → left_actor  (MLP 32→7)  ─┐
  - right_obs (32) → right_actor (MLP 32→7)  ─┤─ concat → 14-dim action
A single 14-dim DiagonalGaussian distribution wraps the combined output.

The left_actor can be initialised from a single-arm rail checkpoint and
optionally frozen so the right arm learns without disturbing the left policy.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class DualArmActor(nn.Module):
    """Actor with two independent MLPs for left/right arms and a shared 14-dim distribution."""

    is_recurrent: bool = False

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
        super().__init__()

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        per_arm_input = self.obs_dim // 2
        per_arm_output = output_dim // 2

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        self.left_actor = MLP(per_arm_input, per_arm_output, hidden_dims, activation)
        self.right_actor = MLP(per_arm_input, per_arm_output, hidden_dims, activation)

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.left_actor)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)

        half = self.obs_dim // 2
        left_out = self.left_actor(latent[:, :half])
        right_out = self.right_actor(latent[:, half:])
        mlp_output = torch.cat([left_out, right_out], dim=-1)

        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        latent = self.obs_normalizer(latent)
        return latent

    # ------------------------------------------------------------------
    # distribution delegation
    # ------------------------------------------------------------------

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    # ------------------------------------------------------------------
    # recurrent stubs
    # ------------------------------------------------------------------

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    # ------------------------------------------------------------------
    # normalisation
    # ------------------------------------------------------------------

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            mlp_obs = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(mlp_obs)

    # ------------------------------------------------------------------
    # pretrained left-arm checkpoint loading
    # ------------------------------------------------------------------

    def load_left_actor(self, checkpoint_path: str) -> None:
        """Load left_actor weights from a single-arm rail checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        left_state = {}
        for k, v in ckpt["actor_state_dict"].items():
            if k.startswith("mlp."):
                left_state[k[4:]] = v  # strip "mlp." prefix
        self.left_actor.load_state_dict(left_state)

    def freeze_left_actor(self) -> None:
        """Freeze the left arm's MLP weights."""
        for p in self.left_actor.parameters():
            p.requires_grad = False

    def load_right_actor(self, checkpoint_path: str) -> None:
        """Load right_actor weights from a single-arm rail checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        right_state = {}
        for k, v in ckpt["actor_state_dict"].items():
            if k.startswith("mlp."):
                right_state[k[4:]] = v  # strip "mlp." prefix
        self.right_actor.load_state_dict(right_state)

    def freeze_right_actor(self) -> None:
        """Freeze the right arm's MLP weights."""
        for p in self.right_actor.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    # ------------------------------------------------------------------
    # export helpers
    # ------------------------------------------------------------------

    def as_jit(self) -> nn.Module:
        return _TorchDualArmActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxDualArmActor(self, verbose)


# -----------------------------------------------------------------------
# TorchScript export wrapper
# -----------------------------------------------------------------------

class _TorchDualArmActor(nn.Module):
    def __init__(self, model: DualArmActor) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.left_actor = copy.deepcopy(model.left_actor)
        self.right_actor = copy.deepcopy(model.right_actor)
        self.half = model.obs_dim // 2
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        left_out = self.left_actor(x[:, :self.half])
        right_out = self.right_actor(x[:, self.half:])
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


class _OnnxDualArmActor(_TorchDualArmActor):
    def __init__(self, model: DualArmActor, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
