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
        

def reinforce(net, optimizer, observations, actions, rewards, lengths, critic=None, critic_optimizer=None, gamma=0.99):
    with torch.no_grad():
        G = torch.zeros_like(rewards)
        running_return = torch.zeros(G.size(0))
        for i in range(G.size(1) - 1, -1, -1):
            running_return = rewards[:, i] + running_return * gamma
            G[:, i] = running_return

    mask = torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)

    critic_loss = torch.tensor(0.0)
    if critic is not None:
        V = critic(observations).squeeze(-1)
        critic_loss = ((V - G.detach()).pow(2) * mask).sum()
        advantage = (G - V.detach())
    else:
        advantage = G

    probs = torch.log_softmax(net(observations), dim=-1)
    action_probs = torch.gather(probs, 2, actions.unsqueeze(2)).squeeze(2)
    J = -(advantage * mask * action_probs).sum() + critic_loss
    J.backward()
    optimizer.step()
    optimizer.zero_grad()
    if critic_optimizer is not None:
        critic_optimizer.step()
        critic_optimizer.zero_grad()
    return rewards.sum(dim=-1).mean(), critic_loss



def record_video(net, env_id="CartPole-v1"):
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
    parser.add_argument("--run-name", type=str, default="reinforce")
    parser.add_argument("--use-baseline", action="store_true")
    args = parser.parse_args()

    critic = None
    critic_optimizer = None
    if args.use_baseline:
        critic = nn.Linear(4, 1)
        critic_optimizer = optim.Adam(critic.parameters(), lr=1e-2)
    

    writer = SummaryWriter(log_dir=f"runs/{args.run_name}")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    net = nn.Linear(4, 2)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    recent_rewards = deque(maxlen=4)

    for i in range(1000):
        trajectories = calculate_trajectories(net, env, 50)
        avg_reward, critic_loss = reinforce(net, optimizer, *merge_trajectories(trajectories), critic=critic, critic_optimizer=critic_optimizer)
        recent_rewards.append(avg_reward.item())
        running_avg = sum(recent_rewards) / len(recent_rewards)
        writer.add_scalar("avg_reward", avg_reward.item(), i)
        writer.add_scalar("critic_loss", critic_loss.item() if critic is not None else 0, i)
        writer.add_scalar("running_avg", running_avg, i)
        print(f"{i} avg_reward={avg_reward:.1f} running_avg={running_avg:.1f}")
        if len(recent_rewards) == 4 and running_avg >= 450:
            print("Solved! Recording video...")
            record_video(net)
            break

    writer.close()








if __name__ == "__main__":
    main()