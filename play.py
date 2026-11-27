"""Play Connect Four against a trained AlphaZero checkpoint.

Usage:
    python play.py <checkpoint.pt> [num_sims]

The architecture (channels / res_blocks) and search params are read from the
config saved inside the checkpoint, so you only pass the file. `num_sims` can be
overridden to make the opponent stronger/weaker than it trained at.
"""

import sys

import numpy as np
import torch

from connect4_env import ConnectFour, COLS
from alphazero import AlphaZeroNet, Node, mcts


def load_network(ckpt_path: str) -> tuple[AlphaZeroNet, dict, object]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    net = AlphaZeroNet(channels=cfg.get("channels", 64), num_res_blocks=cfg.get("res_blocks", 5))
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, cfg, ckpt.get("iteration", "?")


@torch.no_grad()
def net_move(game: ConnectFour, net: AlphaZeroNet, num_sims: int, c: float) -> tuple[int, dict, float]:
    """Run search (no Dirichlet noise) and return (greedy move, visit distribution, value)."""
    root = Node(game)
    mcts(root, net, num_sims, c)
    move = max(root.children, key=lambda a: root.children[a].n)
    total = sum(child.n for child in root.children.values())
    dist = {a: round(root.children[a].n / total, 2) for a in sorted(root.children)}
    value = root.t / root.n if root.n else 0.0   # net's estimate for the side to move (itself)
    return move, dist, value


def ask_human_move(game: ConnectFour) -> int:
    legal = game.legal_moves()
    while True:
        raw = input(f"Your move (legal {legal}): ").strip()
        if not raw.isdigit() or int(raw) not in legal:
            print("  enter one of the legal column numbers.")
            continue
        return int(raw)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python play.py <checkpoint.pt> [num_sims]")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    net, cfg, iteration = load_network(ckpt_path)
    num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.get("num_sims", 200)
    c = cfg.get("c", 1.5)
    print(f"Loaded {ckpt_path} (iteration {iteration}) | "
          f"channels={cfg.get('channels')} res_blocks={cfg.get('res_blocks')} | num_sims={num_sims}")

    human = ""
    while human not in ("x", "o"):
        human = input("Play as x or o? ").strip().lower()

    game = ConnectFour()
    while not game.is_terminal():
        game.render()
        if game.turn == human:
            move = ask_human_move(game)
        else:
            move, dist, value = net_move(game, net, num_sims, c)
            print(f"Net plays {move}  (value {value:+.2f} for itself | visits {dist})")
        game = game.make_move(move)

    game.render()
    result = game.result()  # +1 x, -1 o, 0 draw
    if result == 0:
        print("Draw!")
    elif (result == 1 and human == "x") or (result == -1 and human == "o"):
        print("You win!")
    else:
        print("Net wins!")


if __name__ == "__main__":
    main()
