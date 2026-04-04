from __future__ import annotations

import numpy as np

from .game import ConnectFour


class Node:

    def __init__(self, game: ConnectFour, parent: Node | None = None, action: int | None = None) -> None:
        self.game: ConnectFour = game
        self.children: list[Node] = []
        self.t: float = 0.0
        self.n: int = 0
        self.parent: Node | None = parent
        self.action: int | None = action

    def ucb1(self, c: float) -> float:
        if self.n == 0:
            return float("inf")
        assert self.parent is not None
        return self.t / self.n + c * np.sqrt(np.log(self.parent.n) / self.n)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MCTS:

    def __init__(self, c: float = 2.0, root: Node | None = None) -> None:
        self.root: Node = root if root is not None else Node(ConnectFour())
        self.c: float = c

    def train(self, num_iters: int = 500) -> None:
        for _ in range(num_iters):
            self.train_step()

    def train_step(self) -> None:
        current = self.root
        while not current.is_leaf():
            current = max(current.children, key=lambda node: node.ucb1(c=self.c))
        if current.n == 0 or current.game.is_terminal():
            self.rollout(current)
        else:
            for legal_move in current.game.legal_moves():
                current.children.append(Node(current.game.make_move(legal_move), parent=current, action=legal_move))
            self.rollout(current.children[0])

    def rollout(self, node: Node) -> None:
        game = node.game
        while not game.is_terminal():
            legal_moves = game.legal_moves()
            game = game.make_move(legal_moves[np.random.randint(len(legal_moves))])
        result = game.result()
        reward = -result if node.game.turn == "x" else result
        self.backprop(node, reward)

    def backprop(self, node: Node, reward: float) -> None:
        current: Node | None = node
        while current is not None:
            current.t += reward
            current.n += 1
            reward = -reward
            current = current.parent

    def best_child(self) -> Node:
        return max(self.root.children, key=lambda n: n.n)


def find_child(node: Node, action: int) -> Node | None:
    for child in node.children:
        if child.action == action:
            return child
    return None
