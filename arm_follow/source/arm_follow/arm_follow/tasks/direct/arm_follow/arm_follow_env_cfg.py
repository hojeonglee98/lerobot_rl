from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

from arm_follow.robot import SO_ARM100_ROSCON_CFG  # 👈 from your robots package


@configclass
class ArmFollowEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 5.0

    # match actions/obs to your joints
    action_space: int = 6                # 5 arm + 1 jaw
    observation_space: int = 18          # [q, dq, q_ref] * 6
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
    )

    robot_cfg: ArticulationCfg = SO_ARM100_ROSCON_CFG.replace(
        prim_path="/World/envs/env_.*/SO_ARM100",
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=2.0,
        replicate_physics=True,
    )

    # trajectory + reward etc (as you already set)
    track_in_joint_space: bool = True
    trajectory_period_s: float = 5.0
    trajectory_amplitude: float = 0.3
    rew_scale_tracking: float = 5.0
    rew_scale_action: float = -0.001
    rew_scale_terminal: float = -5.0
    max_joint_pos_err: float = 3.0
    max_action_norm: float = 1.0

