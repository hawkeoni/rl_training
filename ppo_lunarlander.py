import argparse
from collections import deque

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    rewards = [torch.tensor(t.rewards, dtype=torch.float32) for t in trajectories]
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
        
@torch.no_grad()
def generalized_advantage_estimation(rewards, values, mask, gamma=0.99, lambda_=0.98):
    # rewards - [num_trajectories, num_steps]
    values = torch.cat([values, torch.zeros(values.size(0), 1)], dim=1)
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

@torch.no_grad()
def calculate_returns(rewards, gamma):
    G = torch.zeros_like(rewards)
    running_return = torch.zeros(G.size(0))
    for i in range(G.size(1) - 1, -1, -1):
        running_return = rewards[:, i] + running_return * gamma
        G[:, i] = running_return
    return G



def ppo(actor, actor_optimizer, observations, actions, rewards, lengths, critic, critic_optimizer, gamma=0.99):
    mask = torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    values = critic(observations).squeeze(-1)
    advantage = generalized_advantage_estimation(rewards, values, mask, gamma=gamma, lambda_=0.95)
    G = calculate_returns(rewards, gamma)
    with torch.no_grad():
        old_log_probs = torch.log_softmax(actor(observations), dim=-1)

    # mask, rewards, values, advantage [num_trajectories, num_steps]
    # Flatten and keep only valid steps
    valid = mask.flatten()
    obs_flat = observations.flatten(0, 1)[valid]        # [num_valid_steps, obs_dim]
    actions_flat = actions.flatten()[valid]               # [num_valid_steps]
    advantages_flat = advantage.flatten()[valid]          # [num_valid_steps]
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
    returns_flat = G.flatten()[valid]                # [num_valid_steps]
    old_log_probs_flat = old_log_probs.flatten(0, 1)[valid]   # [num_valid_steps, num_actions]


    actor_losses, critic_losses = [], []
    for epoch in range(4):
        actor_losses_local, critic_losses_local = batch_train(actor, actor_optimizer, critic, critic_optimizer, obs_flat, actions_flat, returns_flat, advantages_flat, old_log_probs_flat)
        actor_losses += actor_losses_local
        critic_losses += critic_losses_local
    return rewards.sum(dim=-1).mean(), actor_losses, critic_losses

def batch_train(actor, actor_optimizer, critic, critic_optimizer, obs_flat, actions_flat, returns_flat, advantages_flat, old_log_probs_flat):
    minibatch = 64
    eps = 0.2
    randperm = torch.randperm(obs_flat.size(0))
    actor_losses = []
    critic_losses = []
    for batch_idx in range(0, obs_flat.size(0), minibatch):
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        idx = randperm[batch_idx: batch_idx + minibatch]
        observations = obs_flat[idx]
        old_log_probs  = old_log_probs_flat[idx]
        advantage = advantages_flat[idx]
        # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)


        log_prob_diff = torch.log_softmax(actor(observations), dim=-1) - old_log_probs
        ratio = torch.gather(torch.exp(log_prob_diff), 1, actions_flat[idx].unsqueeze(-1)).squeeze(1)
        L = - torch.min(ratio * advantage, torch.clamp(ratio, 1 - eps, 1 + eps) * advantage).sum()
        actor_losses.append(-L.item())
        critic_loss = F.mse_loss(critic(observations).squeeze(-1), returns_flat[idx])
        critic_losses.append(critic_loss.item())
        total_loss = L + critic_loss

        total_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()
    return actor_losses, critic_losses



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
        avg_reward, actor_losses, critic_losses = ppo(net, optimizer, *merge_trajectories(trajectories), critic=critic, critic_optimizer=critic_optimizer)
        recent_rewards.append(avg_reward.item())
        running_avg = sum(recent_rewards) / len(recent_rewards)
        avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0
        avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0
        writer.add_scalar("avg_reward", avg_reward.item(), i)
        writer.add_scalar("actor_loss", avg_actor_loss, i)
        writer.add_scalar("critic_loss", avg_critic_loss, i)
        writer.add_scalar("running_avg", running_avg, i)
        print(f"{i} avg_reward={avg_reward:.1f} running_avg={running_avg:.1f} actor_loss={avg_actor_loss:.4f} critic_loss={avg_critic_loss:.4f}")
        if len(recent_rewards) == 10 and running_avg >= 200:
            print("Solved! Recording video...")
            record_video(net)
            break

    writer.close()


if __name__ == "__main__":
    main()