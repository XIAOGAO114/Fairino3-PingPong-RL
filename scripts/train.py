# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                    help="Path to a single-arm .pt checkpoint for left-arm MLP weight initialization.")
parser.add_argument("--right_pretrained_checkpoint", type=str, default=None,
                    help="Path to a single-arm .pt checkpoint for right-arm MLP weight initialization.")
parser.add_argument("--unfreeze", action="store_true", default=False,
                    help="Do NOT freeze actors after loading — allow full fine-tuning.")
parser.add_argument("--freeze-left", action="store_true", default=False,
                    help="Freeze left arm MLP even with --unfreeze (for asymmetric fine-tuning).")
parser.add_argument("--right-std", type=float, default=None,
                    help="Override right arm std (indices 7-13) to this value, then freeze it.")
parser.add_argument("--serve-right-only", action="store_true", default=False,
                    help="Always serve toward right arm (prob=1.0).")
parser.add_argument("--ball-speed-scale", type=float, default=1.0,
                    help="Curriculum: scale ball serve velocity (e.g. 0.3 for slow, 1.0 for normal).")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import test_isaac.tasks  # noqa: F401
import test_isaac_dual.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # -- serve direction control --
    if args_cli.serve_right_only:
        isaac_env = env.unwrapped  # ManagerBasedRLEnv
        isaac_env._serve_right_prob = 1.0
        print("[INFO] Serve direction: RIGHT ONLY (100%)")

    # -- curriculum: ball speed scale --
    if args_cli.ball_speed_scale != 1.0:
        isaac_env = env.unwrapped
        isaac_env._ball_speed_scale = args_cli.ball_speed_scale
        print(f"[INFO] Ball speed scale: {args_cli.ball_speed_scale:.2f}x (curriculum mode)")

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # load pretrained single-arm MLP weights via DualArmActor methods
    actor = getattr(runner.alg, "actor", None)
    if args_cli.pretrained_checkpoint and actor is not None:
        print(f"[INFO] Loading left-arm MLP weights from: {args_cli.pretrained_checkpoint}")
        if hasattr(actor, "load_left_actor"):
            actor.load_left_actor(args_cli.pretrained_checkpoint)
            print("[INFO] Left-arm MLP weights loaded.")
        # copy distribution std from pretrained (7-DOF) → dual (14-DOF)
        pretrained = torch.load(args_cli.pretrained_checkpoint, map_location=agent_cfg.device, weights_only=False)
        pretrained_actor = pretrained.get("actor_state_dict", pretrained.get("model_state_dict", {}))
        for std_key in ("std", "distribution.std_param"):
            if std_key in pretrained_actor:
                std_7 = pretrained_actor[std_key]
                if std_7.shape[0] == 7:
                    actor_sd = runner.alg.actor.state_dict()
                    actor_sd["distribution.std_param"] = torch.cat([std_7, std_7])
                    runner.alg.actor.load_state_dict(actor_sd, strict=False)
                    print(f"[INFO] Distribution std expanded 7→14 from pretrained checkpoint.")
                break

    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # freeze / unfreeze actors
    actor = getattr(runner.alg, "actor", None)
    if args_cli.unfreeze:
        # unfreeze both arms for full fine-tuning
        if actor is not None:
            for p in actor.parameters():
                p.requires_grad = True
            # optionally freeze left arm for asymmetric fine-tuning
            if args_cli.freeze_left:
                if hasattr(actor, "freeze_left_actor"):
                    actor.freeze_left_actor()
            # handle std
            if hasattr(actor, "distribution") and hasattr(actor.distribution, "std_param"):
                std_data = actor.distribution.std_param.data  # shape [14]
                # Override right arm std (indices 7-13) if requested
                if args_cli.right_std is not None:
                    std_data[7:] = args_cli.right_std
                    print(f"[INFO] Right arm std (indices 7-13) set to {args_cli.right_std}")
                # Freeze all std
                actor.distribution.std_param.requires_grad = False
                left_std_mean = std_data[:7].mean().item()
                right_std_mean = std_data[7:].mean().item()
                left_status = "FROZEN" if args_cli.freeze_left else "TRAINABLE"
                print(f"[INFO] Actors UNFROZEN, std FROZEN (L-mean={left_std_mean:.4f}, R-mean={right_std_mean:.4f}), left arm {left_status}")
        print("[INFO]: Fine-tuning mode — right arm trainable, left arm " +
              ("frozen" if args_cli.freeze_left else "trainable") + ", std locked.")
    else:
        if actor is not None and hasattr(actor, "freeze_left_actor"):
            actor.freeze_left_actor()
            print("[INFO]: Left-arm actor frozen (pretrained single-arm weights).")

    # load right-arm pretrained weights if provided
    if args_cli.right_pretrained_checkpoint:
        print(f"[INFO] Loading right pretrained MLP weights from: {args_cli.right_pretrained_checkpoint}")
        if actor is not None and hasattr(actor, "load_right_actor"):
            actor.load_right_actor(args_cli.right_pretrained_checkpoint)
            print("[INFO] Right-arm actor weights loaded.")
        if not args_cli.unfreeze:
            if actor is not None and hasattr(actor, "freeze_right_actor"):
                actor.freeze_right_actor()
                print("[INFO]: Right-arm actor frozen (pretrained single-arm weights).")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
