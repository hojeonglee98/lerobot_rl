# lerobot_rl
python scripts/zero_agent.py --task ArmFollow-Direct-v0

python scripts/random_agent.py --task ArmFollow-Direct-v0

python scripts/rsl_rl/train.py --task ArmFollow-Direct-v0 --headless

python scripts/rsl_rl/play.py --task ArmFollow-Direct-v0

Edit /lerobot_rl/arm_follow/source/arm_follow/arm_follow/tasks/direct/arm_follow/arm_follow_env.py to change desired trajectory
