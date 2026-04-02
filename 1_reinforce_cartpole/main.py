import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm


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
    for _ in tqdm(range(n)):
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
        

def reinforce(net, optimizer, observations, actions, rewards, lengths, gamma=0.99):
    with torch.no_grad():
        G = torch.zeros_like(rewards)
        running_return = 0
        for i in range(G.size(0) - 1, -1, -1):
            running_return = rewards[i] + running_return * gamma
            G[i] = running_return
    probs = torch.log_softmax(net(observations), dim=1)
    action_probs = torch.gather(probs, 1, actions.unsqueeze(2)).squeeze(2)
    # action probs [num_trajectories, num_steps]
    # G - [num_trajectories, num_steps]
    # lengths - [num_trajectories]
    mask = torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    J = - (G * mask * action_probs).sum()
    J.backward()
    optimizer.step()
    return rewards.sum(dim=-1).mean()



def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda ep: True)
    net = nn.Linear(4, 2)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for i in range(1000):
        trajectories = calculate_trajectories(net, env, 50)
        print(i, reinforce(net, optimizer, *merge_trajectories(trajectories)))








if __name__ == "__main__":
    main()