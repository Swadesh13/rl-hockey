GAMMA = 0.99
TAU = 0.005
LR = 1e-3
ALPHA = 0.2
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256 * 8
TOTAL_TIMESTEPS = int(1e5)
MAX_STEPS = 251
REWARD_MULTIPLIER = 10
POLICY_NET_ARCH = {
    "pi": [256, 256],  # actor
    "qf": [256, 256],  # critic
}
NOISE = None  # None, pink, brownian
PRIORITIZED_MEMORY = False
TENSORBOARD = "hockey"
