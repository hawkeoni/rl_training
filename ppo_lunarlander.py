import argparse
from collections import deque

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter


class Trajectory:

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
    
    def __len__(self):
        assert len(self.obs) == len(self.actions) == len(self.rewards)
        return len(self.obs)
    

def merge_trajectories(trajectories: list[Trajectory]):
    lengths = [len(t) for t in trajectories]
    observations = [torch.stack(t.obs, dim=0) for t in trajectories]
    observations = pad_sequence(observations, batch_first=True, padding_value=0)
    actions = [torch.tensor(t.actions) for t in trajectories]
    actions = pad_sequence(actions, batch_first=True, padding_value=0)
    rewards = [torch.tensor(t.rewards) for t in trajectories]
    rewards = pad_sequence(rewards, batch_first=True, padding_value=0.0)
    return observations, actions, rewards, lengths



@torch.no_grad()
def calculate_trajectories(actor, env, n) -> list[Trajectory]:
    trajectories = []
    for _ in range(n):
        trajectory = Trajectory()
        obs, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            obs = torch.from_numpy(obs)
            trajectory.obs.append(obs)
            action_distribution = actor(obs.unsqueeze(0)).squeeze(0)
            action = torch.distributions.Categorical(torch.softmax(action_distribution, dim=0)).sample().item()
            trajectory.actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.rewards.append(reward)
        trajectories.append(trajectory)
    return trajectories
        
def generalized_advantage_estimation(rewards, values, mask, gamma=0.99, lambda_=0.98):
    # rewards - [num_trajectories, num_steps]
    # values - [num_trajectories, num_steps + 1]
    gae = torch.zeros_like(rewards)
    deltas = rewards + gamma * values[:, 1:] - values[:, :-1]
    deltas = deltas * mask
    running_return = torch.zeros_like(rewards[:, 0])
    for i in range(rewards.size(1) - 1, -1, -1):
        running_return = running_return * gamma * lambda_ + deltas[:, i]
        gae[:, i] = running_return
    # gae - [num_trajectories, num_steps]
    return gae




def ppo(net, optimizer, observations, actions, rewards, lengths, critic=None, critic_optimizer=None, gamma=0.99):
    mask = torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    values = critic(observations).squeeze(-1)
    values = torch.cat([values, torch.zeros(values.size(0), 1)], dim=1)  # [n_trajectories, T+1]
    advantage = generalized_advantage_estimation(rewards, values, mask, gamma=0.98)
    with torch.no_grad():
        G = torch.zeros_like(rewards)
        running_return = torch.zeros(G.size(0))
        for i in range(G.size(1) - 1, -1, -1):
            running_return = rewards[:, i] + running_return * gamma
            G[:, i] = running_return


    critic_loss = torch.tensor(0.0)
    V = critic(observations).squeeze(-1)
    critic_loss = ((V - G.detach()).pow(2) * mask).sum()
    probs = torch.log_softmax(net(observations), dim=-1)
    action_probs = torch.gather(probs, 2, actions.unsqueeze(2)).squeeze(2)


    J = -(advantage * mask * action_probs).sum() + critic_loss
    J.backward()
    optimizer.step()
    optimizer.zero_grad()
    critic_optimizer.step()
    critic_optimizer.zero_grad()
    return rewards.sum(dim=-1).mean(), critic_loss



def make_actor(obs_dim, num_actions):
    return nn.Sequential(
        nn.Linear(obs_dim, 64),
        nn.ReLU(),
        nn.Linear(64, num_actions),
    )


def make_critic(obs_dim):
    return nn.Sequential(
        nn.Linear(obs_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def record_video(net, env_id="LunarLander-v3"):
    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda ep: True)
    obs, _ = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        obs_t = torch.from_numpy(obs)
        logits = net(obs_t.unsqueeze(0)).squeeze(0)
        action = torch.argmax(logits).item()
        obs, _, terminated, truncated, _ = env.step(action)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="ppo_lunarlander")
    args = parser.parse_args()

    env = gym.make("LunarLander-v3")
    obs_dim = env.observation_space.shape[0]  # 8
    num_actions = env.action_space.n  # 4

    net = make_actor(obs_dim, num_actions)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    critic = make_critic(obs_dim)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    writer = SummaryWriter(log_dir=f"runs/{args.run_name}")
    recent_rewards = deque(maxlen=10)

    for i in range(3000):
        trajectories = calculate_trajectories(net, env, 20)
        avg_reward, critic_loss = ppo(net, optimizer, *merge_trajectories(trajectories), critic=critic, critic_optimizer=critic_optimizer)
        recent_rewards.append(avg_reward.item())
        running_avg = sum(recent_rewards) / len(recent_rewards)
        writer.add_scalar("avg_reward", avg_reward.item(), i)
        writer.add_scalar("critic_loss", critic_loss.item() if critic is not None else 0, i)
        writer.add_scalar("running_avg", running_avg, i)
        print(f"{i} avg_reward={avg_reward:.1f} running_avg={running_avg:.1f}")
        if len(recent_rewards) == 10 and running_avg >= 200:
            print("Solved! Recording video...")
            record_video(net)
            break

    writer.close()


if __name__ == "__main__":
    main()