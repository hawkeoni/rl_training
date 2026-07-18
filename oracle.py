"""Perfect-play Connect 4 solver + objective evals for a trained checkpoint.

The solver is a negamax with alpha-beta pruning, center-first move ordering, and a
transposition table (Connect 4 is solved, so this returns game-theoretic values).
A position's score is >0 if the side to move wins with perfect play, 0 for a draw,
<0 for a loss; larger magnitude = faster win / slower loss, so the argmax move is
the optimal move (and a mate-in-1 always scores highest).

Evals (graded on a fixed, seeded position suite so checkpoints are comparable):
  * mate-in-1  : positions where the side to move can win immediately — does the net play a win?
  * optimal    : arbitrary positions — does the net's move match perfect play (and how often does it blunder)?

Usage:
    python oracle.py <checkpoint.pt> [num_sims]
"""

import sys
import time

import numpy as np

from connect4_env import ConnectFour, COLS
from alphazero import mcts_best_move
from play import load_network

WIDTH, HEIGHT = 7, 6
AREA = WIDTH * HEIGHT
_CENTER_ORDER = sorted(range(WIDTH), key=lambda col: abs(col - WIDTH // 2))  # 3,2,4,1,5,0,6


def stones(game: ConnectFour) -> int:
    return int(np.count_nonzero(game.state))


def immediate_win_move(game: ConnectFour) -> int | None:
    """A move that wins on the spot for the side to move, or None."""
    for m in game.legal_moves():
        if game.make_move(m).result() != 0:
            return m
    return None


class Solver:
    """Exact Connect 4 solver. `score(game)` / `move_scores(game)` from the side-to-move's view."""

    def __init__(self) -> None:
        # key -> (value, flag): flag 0 EXACT, 1 LOWER bound, 2 UPPER bound
        self.tt: dict[bytes, tuple[int, int]] = {}

    def _negamax(self, game: ConnectFour, alpha: int, beta: int) -> int:
        legal = game.legal_moves()
        if not legal:
            return 0  # board full, no winner -> draw

        # Winning move available -> take the fastest win.
        for m in legal:
            child = game.make_move(m)
            if child.result() != 0:
                return (AREA + 1 - stones(child)) // 2

        key = game.state_key()
        orig_alpha = alpha
        entry = self.tt.get(key)
        if entry is not None:
            val, flag = entry
            if flag == 0:
                return val
            if flag == 1:
                alpha = max(alpha, val)
            elif flag == 2:
                beta = min(beta, val)
            if alpha >= beta:
                return val

        # We can't do better than winning on our very next move.
        upper = (AREA - 1 - stones(game)) // 2
        if beta > upper:
            beta = upper
            if alpha >= beta:
                return beta

        best = -AREA
        legal_set = set(legal)
        for m in _CENTER_ORDER:
            if m not in legal_set:
                continue
            val = -self._negamax(game.make_move(m), -beta, -alpha)
            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break

        flag = 2 if best <= orig_alpha else (1 if best >= beta else 0)
        self.tt[key] = (best, flag)
        return best

    def score(self, game: ConnectFour) -> int:
        """Game-theoretic value for the side to move (>0 win, 0 draw, <0 loss)."""
        return self._negamax(game, -AREA, AREA)

    def move_scores(self, game: ConnectFour) -> dict[int, int]:
        """Exact value of each legal move, from the side-to-move's perspective."""
        scores: dict[int, int] = {}
        for m in game.legal_moves():
            child = game.make_move(m)
            if child.result() != 0:                     # winning move
                scores[m] = (AREA + 1 - stones(child)) // 2
            elif not child.legal_moves():               # move fills the board -> draw
                scores[m] = 0
            else:                                        # value = -(opponent's best from child)
                scores[m] = -self.score(child)
        return scores

    def best_moves(self, game: ConnectFour) -> tuple[int, list[int], dict[int, int]]:
        scores = self.move_scores(game)
        best = max(scores.values())
        return best, [m for m, s in scores.items() if s == best], scores


def _sample_positions(rng: np.random.RandomState, predicate, count: int,
                      min_stones: int, max_stones: int) -> list[ConnectFour]:
    """Collect `count` non-terminal positions (one per random game) satisfying `predicate`."""
    out: list[ConnectFour] = []
    while len(out) < count:
        game = ConnectFour()
        target = rng.randint(min_stones, max_stones + 1)
        while not game.is_terminal() and stones(game) < target:
            legal = game.legal_moves()
            game = game.make_move(int(legal[rng.randint(len(legal))]))
        if not game.is_terminal() and predicate(game):
            out.append(game)
    return out


def eval_mate_in_1(network, num_positions: int, num_sims: int, c: float, seed: int = 0) -> float:
    """Fraction of mate-in-1 positions where the net plays an immediately-winning move."""
    rng = np.random.RandomState(seed)
    positions = _sample_positions(rng, lambda g: immediate_win_move(g) is not None,
                                  num_positions, min_stones=6, max_stones=36)
    found = 0
    for game in positions:
        move = mcts_best_move(game, network, num_sims, c)
        if game.make_move(move).result() != 0:
            found += 1
    return found / len(positions)


def eval_optimal(network, num_positions: int, num_sims: int, c: float,
                 min_stones: int, seed: int = 0) -> tuple[float, float]:
    """Returns (optimal_move_rate, blunder_rate) vs perfect play.
    A blunder = the net's move worsens the game-theoretic outcome category (win/draw/loss)."""
    rng = np.random.RandomState(seed)
    solver = Solver()
    positions = _sample_positions(rng, lambda g: True, num_positions,
                                  min_stones=min_stones, max_stones=38)
    optimal = 0
    blunders = 0
    for game in positions:
        best_score, best_set, scores = solver.best_moves(game)
        move = mcts_best_move(game, network, num_sims, c)
        if move in best_set:
            optimal += 1
        # outcome category: sign of the score (+1 win / 0 draw / -1 loss)
        if np.sign(scores[move]) < np.sign(best_score):
            blunders += 1
    n = len(positions)
    return optimal / n, blunders / n


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python oracle.py <checkpoint.pt> [num_sims]")
        sys.exit(1)
    ckpt_path = sys.argv[1]
    network, cfg, iteration = load_network(ckpt_path)
    num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.get("num_sims", 200)
    c = cfg.get("c", 1.5)
    print(f"Grading {ckpt_path} (iteration {iteration}) at num_sims={num_sims}")

    t0 = time.time()
    m1 = eval_mate_in_1(network, num_positions=100, num_sims=num_sims, c=c)
    print(f"  mate-in-1 found:      {m1:.1%}  ({time.time() - t0:.1f}s)")

    # NOTE: the optimal-move eval uses the exact solver, which is still the slow
    # numpy version and hangs on shallow positions. Re-enable once the bitboard
    # solver lands. For now, mate-in-1 is the objective check.
    # opt, blunder = eval_optimal(network, num_positions=100, num_sims=num_sims, c=c, min_stones=24)
    # print(f"  optimal-move match:   {opt:.1%}")
    # print(f"  blunder rate:         {blunder:.1%}")


if __name__ == "__main__":
    main()
