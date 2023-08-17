import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.utils import set_random_seed

import torch

# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html


def make_env(env_id: str, rank: int, seed: int = 0, render_mode: str = "human"):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    using_gpu = torch.cuda.is_available()
    if not using_gpu:
        print("Not using GPU. Aborting!")
        exit(-1)
    print(f"GPU device count: {torch.cuda.device_count()}")
    curr_device = torch.cuda.current_device()
    print(f"Current device: {curr_device}")
    print(f"Current device name: {torch.cuda.get_device_name(curr_device)}")
    print(f"Current device capability: {torch.cuda.get_device_capability(curr_device)}")

    env_id = "CartPole-v1"
    num_env = 16  # Number of processes to use
    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, i, render_mode="rgb_array") for i in range(num_env)])
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    model.learn(total_timesteps=90_000, progress_bar=True)
    model.save("testmodel")
    del model



    # vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    vec_env2 = make_vec_env(env_id, n_envs=4, seed=0)
    # vec_envs = VecFrameStack(vec_env, n_stack=4)
    model = PPO.load("testmodel", vec_env2)
    obs = vec_env2.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env2.step(action)
        vec_env2.render("human")
