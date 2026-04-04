import random
import argparse
import pickle
from pathlib import Path
import torch
import numpy as np
from dataclasses import dataclass

from alphazero_connect4.config import TrainConfig, PlayConfig
from alphazero_connect4.model import AlphaZeroNet
from alphazero_connect4.game import ConnectFour, COLS
from alphazero_connect4.mcts import MCTS, Node, find_child



class ReplayElement:
    def __init__(self, state: torch.Tensor, probabilities: torch.Tensor, win: int | None) -> None:
        self.state = state
        self.probabilities = probabilities
        self.win = win

def merge_replay_buffer(replay_buffer: list[ReplayElement]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.stack([r.state for r in replay_buffer])
    probabilities = torch.stack([r.probabilities for r in replay_buffer])
    winners = torch.tensor([r.win for r in replay_buffer], dtype=torch.float32).unsqueeze(1)
    return states, probabilities, winners

class Node:

    def __init__(self, game: ConnectFour, parent: Node | None = None, action: int | None = None, prior_probability: float | None = None) -> None:
        self.game: ConnectFour = game
        self.children: dict[int, Node] = {}
        self.t: float = 0.0
        self.n: int = 0
        self.parent: Node | None = parent
        self.action: int | None = action
        self.prior_probability = prior_probability
        self.children_probabilities = []

    def puct(self, c: float) -> float:
        q = self.t / self.n if self.n else 0
        return q + self.prior_probability * c * (self.parent.n ** 0.5) / (1 + self.n)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __del__(self) -> None:
        for child in self.children.values():
            child.parent = None
            child.__del__()
        self.children.clear()
        self.parent = None

    def to_replay_node(self) -> ReplayElement:
        children_probabilies = torch.zeros(7)
        for move in range(7):
            if move in self.children:
                children_probabilies[move] = self.children[move].n
        children_probabilies = children_probabilies / children_probabilies.sum()
        return ReplayElement(self.game.to_tensor(), children_probabilies, None)



def mcts(root: Node, network: AlphaZeroNet, num_steps: int, c: float):
    for _ in range(num_steps):
        current = root
        # we pick children using PUCT until we reach a nod that has never been expanded
        while not current.is_leaf():
            current = max(current.children.values(), key=lambda node: node.puct(c))

        # we are at a leaf node, evaluate this position and expand it

        # Evaluating position
        state = current.game.to_tensor()
        legal_moves = current.game.legal_moves()
        action_logits, value = network(state.unsqueeze(0))
        legal_moves_mask = torch.BoolTensor([0] * 7)
        legal_moves_mask[legal_moves] = 1

        action_logits = action_logits.squeeze(0).masked_fill(~legal_moves_mask, -1e9)
        action_probs = torch.softmax(action_logits, dim=-1)
        current.children_probabilities = action_probs
        value = value.squeeze(0)
        current.t = value.item()
        current.n = 1
        # Expanding children
        for move in legal_moves:
            current.children[move] = Node(current.game.make_move(move), current, move, action_probs[move].item())
        
        # Backup
        current = current.parent
        value = - value
        while current is not None:
            current.n += 1
            current.t += value
            value = - value
            current = current.parent


@torch.no_grad()
def simulate_game(network: AlphaZeroNet, num_mcts_sims: int, c: float) -> list[ReplayElement]:
    game = ConnectFour()
    root = Node(game)
    replay_elements = []
    while not game.is_terminal():
        mcts(root, network, num_mcts_sims, c)
        replay_elements.append(root.to_replay_node())
        move = max(root.children.keys(), key=lambda m: root.children[m].n)
        game = game.make_move(move)
        new_root = root.children.pop(move)
        new_root.parent = None
        del root
        root = new_root
    
    winner = game.result()
    for replay_element in replay_elements:
        replay_element.win = winner
        winner = -winner
    return replay_elements


def train_model(network: AlphaZeroNet, optimizer, states, probabilities, winners, batch_size: int):
    N = states.size(0)
    randperm = torch.randperm(N)
    ce_loss = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    policy_losses = []
    value_losses = []

    for start in range(0, N, batch_size):
        idx = randperm[start: start + batch_size]
        iter_states = states[idx]
        iter_probs = probabilities[idx]
        iter_winners = winners[idx]
        probs, values = network(iter_states)

        policy_loss = ce_loss(probs, iter_probs)
        value_loss = mse(values, iter_winners)
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return policy_losses, value_losses



def train(config: TrainConfig) -> None:
    save_folder = Path(config.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    network = AlphaZeroNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        print(f"=== Epoch {epoch + 1}/{config.num_epochs} ===")
        replay_buffer = []
        for _ in range(config.num_games_per_epoch):
            replay_buffer += simulate_game(network, config.num_mcts_sims, config.c)
        for _ in range(config.num_sub_epochs):
            states, probabilities, winners = merge_replay_buffer(replay_buffer)
            policy_losses, value_losses = train_model(network, optimizer, states, probabilities, winners, batch_size=config.batch_size)


def play(config: PlayConfig) -> None:
    game = ConnectFour()
    with open(config.model_path, "rb") as f:
        root: Node = pickle.load(f)
    mcts = MCTS(c=2.0, root=root)
    print(f"Loaded model from {config.model_path}")

    human_side: str = input("Play as x or o? ").strip().lower()
    assert human_side in ("x", "o")

    while not game.is_terminal():
        game.render()
        if game.turn == human_side:
            legal = game.legal_moves()
            print(f"Legal columns (0-{COLS - 1}): {legal}")
            move = int(input("Your move: "))
            assert move in legal
        else:
            mcts.train(num_iters=config.num_mcts_sims)
            best = mcts.best_child()
            move = best.action
            assert move is not None
            print(f"MCTS plays {move}  (visits: {best.n}, value: {best.t / best.n:.3f})")

        game = game.make_move(move)
        child = find_child(mcts.root, move)
        if child is not None:
            child.parent = None
            mcts = MCTS(c=mcts.c, root=child)
        else:
            mcts = MCTS(c=mcts.c, root=Node(game))

    game.render()
    result = game.result()
    if result == 0:
        print("Draw!")
    elif (result == 1 and human_side == "x") or (result == -1 and human_side == "o"):
        print("You win!")
    else:
        print("MCTS wins!")


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero-style MCTS Connect Four")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train via self-play")
    train_parser.add_argument("--config", type=str, required=False, help="Path to play config JSON", default=None)

    play_parser = subparsers.add_parser("play", help="Play against the agent")
    play_parser.add_argument("--config", type=str, required=False, help="Path to play config JSON", default=None)

    args = parser.parse_args()

    if args.command == "train":
        if args.config:
            with open(args.config) as f:
                config = TrainConfig.model_validate_json(f.read())
        else:
            config = TrainConfig()
        train(config)
    elif args.command == "play":
        if args.config:
            with open(args.config) as f:
                play_config = PlayConfig.model_validate_json(f.read())
        else:
            play_config = PlayConfig(model_path="model.pt")
        play(play_config)


if __name__ == "__main__":
    main()
