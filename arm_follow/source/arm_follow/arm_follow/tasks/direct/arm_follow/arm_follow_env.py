# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .arm_follow_env_cfg import ArmFollowEnvCfg


class ArmFollowEnv(DirectRLEnv):
    """Robot arm trajectory-following environment (joint-space)."""

    cfg: ArmFollowEnvCfg

    def __init__(self, cfg: ArmFollowEnvCfg, render_mode: str | None = None, **kwargs):
        # This will call _setup_scene() internally
        super().__init__(cfg, render_mode, **kwargs)

        self.num_actions = self.cfg.action_space

        # Effective dt for RL (actions applied every `decimation` sim steps)
        self.rl_dt = self.cfg.sim.dt * self.cfg.decimation
        # Buffer for last actions (for action penalty)
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

    # --------------------------------------------------------------------- #
    # Scene setup
    # --------------------------------------------------------------------- #

    def _setup_scene(self):
        """Create robot, ground, lights, and cloned environments."""
        # Robot from cfg
        self.robot = Articulation(self.cfg.robot_cfg)

        # Ground plane (optional but nice)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Add articulation to scene before cloning
        self.scene.articulations["robot"] = self.robot

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # CPU collision filtering (same as template)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Simple dome light
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    # --------------------------------------------------------------------- #
    # RL hooks
    # --------------------------------------------------------------------- #

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions; DirectRLEnv will call _apply_action each sim step."""
        if actions is None:
            return

        # clip to allowed range
        actions = torch.clamp(actions, -self.cfg.max_action_norm, self.cfg.max_action_norm)
        self.actions = actions

    def _apply_action(self) -> None:
        """Apply joint commands based on current actions."""
        # Example: interpret actions as joint velocity targets
        # You can switch to position or effort if it matches your actuators better.
        target_vel = self.actions  # [num_envs, num_actions]

        # Broadcast to all envs on the selected DOFs
        self.robot.set_joint_velocity_target(
            target_vel,
            joint_ids=self.control_dof_indices,
        )

    def _get_observations(self) -> dict:
        """Observation: [q, dq, q_ref] for controlled joints."""
        q = self.joint_pos[:, self.control_dof_indices]   # [N, A]
        dq = self.joint_vel[:, self.control_dof_indices]  # [N, A]
        q_ref = self._get_reference_trajectory()          # [N, A]

        obs = torch.cat([q, dq, q_ref], dim=-1)           # [N, 3A]
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Reward based on tracking error and action regularization."""
        q = self.joint_pos[:, self.control_dof_indices]
        q_ref = self._get_reference_trajectory()

        # tracking error
        err = q - q_ref
        tracking_cost = torch.sum(err * err, dim=-1)

        # action cost
        act_cost = torch.sum(self.actions * self.actions, dim=-1)

        rew = (
            -self.cfg.rew_scale_tracking * tracking_cost
            + self.cfg.rew_scale_action * act_cost
        )

        # big negative if already marked terminated (optional, via reset_terminated)
        if hasattr(self, "reset_terminated"):
            rew = rew + self.cfg.rew_scale_terminal * self.reset_terminated.float()

        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            reset_terminated: envs that hit failure (bool)
            reset_time_out: envs that ended due to episode length (bool)
        """
        q = self.joint_pos[:, self.control_dof_indices]
        q_ref = self._get_reference_trajectory()

        # tracking error based termination
        err = torch.abs(q - q_ref)
        too_far = torch.any(err > self.cfg.max_joint_pos_err, dim=-1)

        # timeout from DirectRLEnv buffers
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        reset_terminated = too_far
        reset_time_out = time_out

        return reset_terminated, reset_time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset selected environments to default joint states & origins."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)
	
        if not hasattr(self, "control_dof_indices"):
            # Cache views into joint state tensors for speed
            # These must be set FIRST!
            self.joint_pos = self.robot.data.joint_pos
            self.joint_vel = self.robot.data.joint_vel
            
            # Use the tensor shape to find the total number of DOFs (CORRECTED LINE)
            num_dof = self.joint_pos.shape[-1]
            
            # Create the indices for the controlled DOFs
            dof_indices = torch.arange(num_dof, device=self.device)
            self.control_dof_indices = dof_indices[: self.num_actions]
            
        # default joint and root states
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        root_state = self.robot.data.default_root_state[env_ids]
        root_state[:, :3] = self.scene.env_origins[env_ids]

        # write to sim
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # clear counters and actions
        self.episode_length_buf[env_ids] = 0
        self.actions[env_ids] = 0.0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_reference_trajectory(self) -> torch.Tensor:
        """Joint-space sinusoidal reference trajectory for each env and joint."""
        if not self.cfg.track_in_joint_space:
            raise NotImplementedError("Only joint-space tracking implemented.")

        # time per env (episode_length_buf increments every RL step)
        t = self.episode_length_buf.to(dtype=torch.float32, device=self.device) * self.rl_dt  # [N]
        omega = 2.0 * math.pi / self.cfg.trajectory_period_s

        phase = (omega * t).unsqueeze(-1)                     # [N, 1]
        base = torch.sin(phase)                              # [N, 1]

        # Same sine on all joints as a starting point
        q_ref = self.cfg.trajectory_amplitude * base.repeat(1, self.num_actions)  # [N, A]

        return q_ref
