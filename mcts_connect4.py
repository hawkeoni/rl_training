import argparse

import numpy as np

from connect4_env import ConnectFour, ROWS, COLS, CONNECT


class Node:

    def __init__(self, game: ConnectFour, parent: "Node" = None, action: int = None):
        self.game = game
        self.children = []
        self.t = 0
        self.n = 0
        self.parent = parent
        self.action = action

    def ucb1(self, c):
        if self.n == 0:
            return float("inf")
        return self.t / self.n + c * np.sqrt(np.log(self.parent.n) / self.n)
    
    def is_leaf(self):
        return len(self.children) == 0



class MCTS:

    def __init__(self, c=2.0, root: Node = None):
        self.root = root if root is not None else Node(ConnectFour())
        self.c = c

    def train(self, num_iters: int = 500):
        for _ in range(num_iters):
            self.train_step()

    def train_step(self):
        current = self.root
        while not current.is_leaf():
            current = max(current.children, key=lambda node: node.ucb1(c=self.c))
        if current.n == 0 or current.game.is_terminal():
            self.rollout(current)
        else:
            for legal_move in current.game.legal_moves():
                current.children.append(Node(current.game.make_move(legal_move), parent=current, action=legal_move))
            self.rollout(current.children[0])
    
    def rollout(self, node: Node):
        game = node.game
        while not game.is_terminal():
            legal_moves = game.legal_moves()
            game = game.make_move(legal_moves[np.random.randint(len(legal_moves))])
        result = game.result()
        # Convert absolute result to parent's perspective for correct backprop
        reward = -result if node.game.turn == "x" else result
        self.backprop(node, reward)

    def backprop(self, node: Node, reward: float):
        while node is not None:
            node.t += reward
            node.n += 1
            reward = -reward
            node = node.parent

    def best_child(self):
        return max(self.root.children, key=lambda n: n.n)


def find_child(node: Node, action: int):
    for child in node.children:
        if child.action == action:
            return child
    return None


def random_move(game: ConnectFour) -> int:
    legal = game.legal_moves()
    return legal[np.random.randint(len(legal))]


def play_game(iters: int, c: float, mcts_side: str, human_side: str = None, verbose: bool = True) -> int:
    """Play one game. The MCTS agent plays `mcts_side`; the other side is a human
    (if `human_side` is set) or a random opponent. Returns +1 (x wins), -1 (o wins), 0 (draw)."""
    game = ConnectFour()
    mcts = MCTS(c=c, root=Node(game))

    while not game.is_terminal():
        if verbose:
            game.render()
        if game.turn == mcts_side:
            mcts.train(num_iters=iters)
            best = mcts.best_child()
            move = best.action
            if verbose:
                print(f"MCTS ({mcts_side}) plays {move}  (visits: {best.n}, value: {best.t / best.n:.3f})")
        elif human_side is not None:
            legal = game.legal_moves()
            print(f"Legal columns (0-{COLS - 1}): {legal}")
            move = int(input("Your move: "))
            assert move in legal
        else:
            move = random_move(game)
            if verbose:
                print(f"Random ({game.turn}) plays {move}")

        game = game.make_move(move)
        # Reuse subtree if the child was previously explored
        child = find_child(mcts.root, move)
        if child is not None:
            child.parent = None
            mcts = MCTS(c=mcts.c, root=child)
        else:
            mcts = MCTS(c=mcts.c, root=Node(game))

    if verbose:
        game.render()
    return game.result()


def main():
    parser = argparse.ArgumentParser(description="Play Connect Four against an MCTS agent.")
    parser.add_argument("--iters", type=int, default=10000, help="MCTS iterations per move")
    parser.add_argument("--c", type=float, default=2.0, help="UCB1 exploration constant")
    parser.add_argument("--opponent", choices=["human", "random"], default="human",
                        help="Who plays against the MCTS agent")
    parser.add_argument("--games", type=int, default=1,
                        help="Number of games to play (random opponent only)")
    parser.add_argument("--mcts-side", choices=["x", "o"], default=None,
                        help="Side the MCTS agent plays vs a random opponent (default: alternate each game)")
    args = parser.parse_args()

    if args.opponent == "human":
        human_side = input("Play as x or o? ").strip().lower()
        assert human_side in ("x", "o")
        mcts_side = "o" if human_side == "x" else "x"
        result = play_game(args.iters, args.c, mcts_side, human_side=human_side, verbose=True)
        if result == 0:
            print("Draw!")
        elif (result == 1 and human_side == "x") or (result == -1 and human_side == "o"):
            print("You win!")
        else:
            print("MCTS wins!")
        return

    # Random opponent (possibly many games)
    wins = draws = losses = 0
    for i in range(args.games):
        mcts_side = args.mcts_side if args.mcts_side is not None else ("x" if i % 2 == 0 else "o")
        verbose = args.games == 1
        result = play_game(args.iters, args.c, mcts_side, human_side=None, verbose=verbose)
        mcts_won = (result == 1 and mcts_side == "x") or (result == -1 and mcts_side == "o")
        if result == 0:
            draws += 1
            outcome = "draw"
        elif mcts_won:
            wins += 1
            outcome = "MCTS win"
        else:
            losses += 1
            outcome = "MCTS loss"
        print(f"Game {i + 1}/{args.games}: {outcome}  (MCTS={mcts_side})")

    print(f"\nMCTS vs random over {args.games} games: {wins}W {draws}D {losses}L")


if __name__ == "__main__":
    main()