


import random
import sys
import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.tensorboard import SummaryWriter

from connect4_env import ConnectFour, ROWS, COLS, CONNECT


# ============================================================================
# Config
# ============================================================================


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Search / self-play
    num_sims: int = 200                 # MCTS simulations per move
    c: float = 1.5                      # PUCT exploration constant
    dirichlet_alpha: float = 1.0        # Dirichlet noise alpha added to root priors
    dirichlet_eps: float = 0.25         # Weight of Dirichlet noise mixed into root priors
    temp_move_threshold: int = 10       # Sample moves ~ visit counts for this many opening plies, then greedy

    # Network
    channels: int = 64                  # Conv trunk channels
    res_blocks: int = 5                 # Number of residual blocks

    # Training loop
    iterations: int = 100               # Outer iterations (self-play + train)
    games_per_iter: int = 100           # Self-play games per iteration
    train_steps_per_iter: int = 1000    # Gradient steps per iteration
    batch_size: int = 128               # Minibatch size
    lr: float = 1e-3                    # Adam learning rate
    weight_decay: float = 1e-4          # Optimizer weight decay
    replay_buffer_size: int = 100_000   # Max positions kept in the replay buffer

    # Evaluation
    eval_interval: int = 5              # Evaluate every N iterations
    eval_games: int = 40                # Games per evaluation
    eval_random_opening: int = 4        # Random opening plies per eval game (diversifies openings)

    # Checkpointing / misc
    save_dir: str = "checkpoints"       # Directory for checkpoints and logs
    save_interval: int = 5              # Checkpoint every N iterations
    autoresume: bool = True             # Resume from the latest checkpoint in save_dir if one exists
    seed: int = 0                       # Random seed
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str = "config.json") -> Config:
    """Load a Config from JSON. Missing keys fall back to the defaults above;
    unknown keys raise. If the file doesn't exist, use all defaults."""
    p = Path(path)
    if p.exists():
        print(f"Loading config from {p}")
        return Config.model_validate_json(p.read_text())
    print(f"No config at {p}; using defaults")
    return Config()

# The ConnectFour env (with to_tensor board encoding) lives in connect4_env.py and
# is shared with mcts_connect4.py. Treat it as fixed; everything below is yours.


# ============================================================================
# Network
# ============================================================================


class ResBlock(nn.Module):
    """3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> skip-add -> ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNet(nn.Module):
    """Two-headed net: board tensor -> (policy logits over COLS moves, value in [-1, 1]).

    The value is from the perspective of the side to move (+1 = current player is
    winning). Return raw policy *logits* (apply softmax / legal-move masking in the
    MCTS, not here)."""

    def __init__(self, channels: int = 64, num_res_blocks: int = 5) -> None:
        super().__init__()
        # Body: initial 3x3 conv -> BN -> ReLU, then a stack of residual blocks.
        self.conv_in = nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList(ResBlock(channels) for _ in range(num_res_blocks))

        # Policy head: 1x1 conv -> 2 filters -> BN -> ReLU -> flatten (2*6*7) -> dense to COLS.
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * ROWS * COLS, COLS)

        # Value head: 1x1 conv -> 1 filter -> BN -> ReLU -> flatten (42) -> dense 64 -> ReLU -> dense 1 -> tanh.
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(ROWS * COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, 2, ROWS, COLS) -> (policy_logits (batch, COLS), value (batch, 1))
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


# ============================================================================
# MCTS
# ============================================================================


class Node:
    """One node in the search tree = one game position.

    Track: visit count n, value sum t (so mean value = t / n), the prior
    probability of the move that led here (from the parent's policy), and children
    keyed by action."""

    def __init__(self, game: ConnectFour, parent: "Node" = None,
                 action: int | None = None, prior: float = 0.0) -> None:
        self.game = game
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: dict[int, Node] = {}
        self.n: int = 0
        self.t: float = 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def puct(self, c: float) -> float:
        """PUCT score used to select a child during search:
            Q + c * prior * sqrt(parent.n) / (1 + n)
        where Q is this node's mean value (0 if unvisited)."""
        q = self.t / self.n if self.n > 0 else 0.0
        return -q + c * self.prior * (self.parent.n ** 0.5) / (1 + self.n)
    
    def move_probs(self) -> list[float]:
        res = []
        for move in range(7):
            if move in self.children:
                res.append(self.children[move].n)
            else:
                res.append(0)
        s = sum(res)
        assert s > 0
        for i in range(len(res)):
            res[i] /= s
        return res


def add_dirichlet_noise(node: Node, alpha: float, eps: float) -> None:
    """Mix Dirichlet noise into the root's child priors: prior <- (1-eps)*prior + eps*noise.
    Applied once, at the root, to encourage exploration during self-play."""
    moves = list(node.children.keys())
    noise = np.random.dirichlet([alpha] * len(moves))
    for move, n in zip(moves, noise):
        node.children[move].prior = (1 - eps) * float(node.children[move].prior) + eps * float(n)


@torch.no_grad()
def mcts(root: Node, network: AlphaZeroNet, num_sims: int, c: float,
         dirichlet_alpha: float = 0.0, dirichlet_eps: float = 0.0) -> None:
    """Run `num_sims` simulations, mutating the tree in place. Each simulation:
      1. Select: from root, follow the highest-PUCT child until reaching a leaf.
      2. Evaluate + expand: run the network on the leaf; use value for backup and
         the (legal-move-masked) policy as priors to create the leaf's children.
         For terminal leaves, use the true game result instead of the network.
      3. Backup: propagate the value up to the root, flipping sign each ply
         (the value is always from the perspective of the node's side to move).
    If `dirichlet_eps > 0`, Dirichlet noise is mixed into the root's child priors
    (once, when the root is first expanded) to encourage exploration in self-play.
    Evaluation should leave `dirichlet_eps=0` for deterministic strongest play."""
    device = next(network.parameters()).device

    for _ in range(num_sims):
        node = root
        # select
        while not node.is_leaf():
            node = max(node.children.values(), key=lambda n: n.puct(c))
        # evaluate + expand
        if node.game.is_terminal():
            z = node.game.result()
            if node.game.turn == "o":
                z = -z
            node.t += z
            node.n += 1
        else:
            # Inside this block every tensor factory (to_tensor, the mask index, ...)
            # is created on `device` automatically — no manual .to() needed.
            with torch.device(device):
                p_logits, v = network(node.game.to_tensor())
                illegal_moves = node.game.illegal_moves()
                p_logits = p_logits.index_fill(1, torch.tensor(illegal_moves, dtype=torch.long), -1e6)
                p_probs = torch.softmax(p_logits, dim=1)
            legal_moves = node.game.legal_moves()
            for move in legal_moves:
                node.children[move] = Node(node.game.make_move(move), node, move, p_probs[0, move].item())
            z = v.item()
            node.t = z
            node.n = 1
            # Root exploration noise (self-play only): applied once, at first root expansion.
            if node is root and dirichlet_eps > 0:
                add_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps)
        # backprop
        while node.parent is not None:
            node = node.parent
            z = -z
            node.n += 1
            node.t +=  z


# ============================================================================
# Self-play
# ============================================================================


@dataclass
class ReplayElement:
    move_probs: list[float]
    state: torch.Tensor
    z: float


def self_play(network: AlphaZeroNet, num_sims: int, c: float, temp_move_threshold: int,
              dirichlet_alpha: float = 0.0, dirichlet_eps: float = 0.0) -> list[ReplayElement]:
    """Play one full game where both sides are driven by MCTS on `network`.

    At each move: run MCTS, record (board_tensor, visit-count policy over COLS),
    then pick a move (sample early for exploration, argmax later). After the game
    ends, label every recorded position with the outcome z from that position's
    side-to-move perspective (+1 win / -1 loss / 0 draw), flipping sign each ply.

    Returns a list of (state, policy_target, z) training examples."""
    game = ConnectFour()
    history = []
    move_number = 0
    while not game.is_terminal():
        node = Node(game)
        mcts(node, network, num_sims, c, dirichlet_alpha, dirichlet_eps)
        move_probs = node.move_probs()
        history.append(ReplayElement(move_probs, node.game.to_tensor(), 0))
        if move_number < temp_move_threshold:
            move = np.random.choice(COLS, p=move_probs)
        else:
            move = np.argmax(move_probs)
        move_number += 1
        game = node.game.make_move(move)
    z = game.result()
    if node.game.turn == "o":
        z = -z
    for element in reversed(history):
        element.z = z
        z = -z
    return history
        


# ============================================================================
# Training
# ============================================================================


def train_step(network: AlphaZeroNet, optimizer: torch.optim.Optimizer,
               batch: list[ReplayElement]) -> tuple[float, float]:
    """One gradient step on a batch of (state, policy_target, z).
    Loss = cross-entropy(policy_logits, policy_target) + MSE(value, z).
    Returns (policy_loss, value_loss) as floats."""
    # The replay buffer holds CPU tensors; move each assembled batch to the net's
    # device. (A torch.device() context wouldn't help here: cat/vstack over existing
    # CPU tensors keep their device — only tensor *factories* honor the context.)
    device = next(network.parameters()).device
    x = torch.cat([b.state for b in batch], dim=0).to(device)
    y_p = torch.vstack([torch.Tensor(b.move_probs) for b in batch]).to(device)
    y_v = torch.tensor([b.z for b in batch], dtype=torch.float32).to(device)
    p, v = network(x)
    v = v.squeeze(1)
    loss_p = F.cross_entropy(p, y_p)
    loss_v = F.mse_loss(v, y_v)
    loss = loss_p + loss_v
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss_p.item(), loss_v.item()


def find_latest_checkpoint(save_dir: Path) -> Path | None:
    """Return the checkpoint with the highest iteration in save_dir, or None."""
    ckpts = list(save_dir.glob("checkpoint_iter_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))


def train(config: Config) -> None:
    """Outer loop: repeatedly self-play to fill a replay buffer, train on samples
    from it, checkpoint, and periodically evaluate vs the previous snapshot and vs
    a random opponent. Wires together self_play, train_step, and evaluate.

    Note on devices: the network is moved to `config.device` here; self_play, mcts,
    and train_step can recover it via `next(network.parameters()).device` and move
    board tensors / batches onto it."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(config.device)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))

    network = AlphaZeroNet(channels=config.channels, num_res_blocks=config.res_blocks).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    replay_buffer: deque = deque(maxlen=config.replay_buffer_size)
    prev_network: AlphaZeroNet | None = None  # snapshot from the last eval, for "vs previous"
    global_step = 0
    start_iteration = 1

    # ---- Auto-resume from the latest checkpoint (+ replay buffer) in save_dir ----
    latest = find_latest_checkpoint(save_dir) if config.autoresume else None
    if latest is not None:
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        network.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = ckpt["iteration"] + 1
        global_step = ckpt["iteration"] * config.train_steps_per_iter  # keep TB x-axis continuous
        buf_path = save_dir / "replay_buffer.pt"
        if buf_path.exists():
            replay_buffer = deque(torch.load(buf_path, weights_only=False), maxlen=config.replay_buffer_size)
        print(f"Resumed from {latest} (iteration {ckpt['iteration']}) | "
              f"buffer {len(replay_buffer)} | continuing at iteration {start_iteration}")

    for iteration in range(start_iteration, config.iterations + 1):
        print(f"=== Iteration {iteration}/{config.iterations} ===")

        # ---- Self-play ----
        network.eval()
        t0 = time.time()
        new_examples = 0
        for _ in tqdm(range(config.games_per_iter), desc="Self-play"):
            examples = self_play(network, config.num_sims, config.c, config.temp_move_threshold,
                                 config.dirichlet_alpha, config.dirichlet_eps)
            replay_buffer.extend(examples)
            new_examples += len(examples)
        selfplay_time = time.time() - t0
        writer.add_scalar("selfplay/new_examples", new_examples, iteration)
        writer.add_scalar("selfplay/buffer_size", len(replay_buffer), iteration)
        writer.add_scalar("time/selfplay_sec", selfplay_time, iteration)

        # ---- Training ----
        network.train()
        t0 = time.time()
        policy_sum = value_sum = 0.0
        steps = 0
        if len(replay_buffer) >= config.batch_size:
            for _ in tqdm(range(config.train_steps_per_iter), desc="Train"):
                batch = random.sample(replay_buffer, config.batch_size)
                policy_loss, value_loss = train_step(network, optimizer, batch)
                writer.add_scalar("loss/policy", policy_loss, global_step)
                writer.add_scalar("loss/value", value_loss, global_step)
                policy_sum += policy_loss
                value_sum += value_loss
                steps += 1
                global_step += 1
        else:
            print(f"  (skipping training: buffer {len(replay_buffer)} < batch size {config.batch_size})")
        train_time = time.time() - t0
        writer.add_scalar("time/train_sec", train_time, iteration)
        if steps:
            writer.add_scalar("loss/policy_avg", policy_sum / steps, iteration)
            writer.add_scalar("loss/value_avg", value_sum / steps, iteration)
            print(f"  self-play {selfplay_time:.1f}s ({new_examples} ex) | "
                  f"train {train_time:.1f}s | policy {policy_sum / steps:.4f} value {value_sum / steps:.4f}")

        # ---- Evaluation ----
        if iteration % config.eval_interval == 0:
            wr_random, rec_random = evaluate(network, None, config.eval_games, config.num_sims, config.c,
                                             config.eval_random_opening)
            writer.add_scalar("eval/winrate_vs_random", wr_random, iteration)
            print(f"  vs random:   {wr_random:.1%}  (W/D/L {rec_random[0]}/{rec_random[1]}/{rec_random[2]})")

            if prev_network is not None:
                wr_prev, rec_prev = evaluate(network, prev_network, config.eval_games, config.num_sims, config.c,
                                             config.eval_random_opening)
                writer.add_scalar("eval/winrate_vs_previous", wr_prev, iteration)
                print(f"  vs previous: {wr_prev:.1%}  (W/D/L {rec_prev[0]}/{rec_prev[1]}/{rec_prev[2]})")

            # Snapshot current weights as the "previous" baseline for the next eval.
            prev_network = AlphaZeroNet(channels=config.channels, num_res_blocks=config.res_blocks).to(device)
            prev_network.load_state_dict(network.state_dict())
            prev_network.eval()

        # ---- Checkpoint ----
        if iteration % config.save_interval == 0:
            ckpt_path = save_dir / f"checkpoint_iter_{iteration}.pt"
            torch.save({
                "model": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "config": config.model_dump(),
            }, ckpt_path)
            # Dump the replay buffer (single rolling file, synced to the latest checkpoint).
            torch.save(list(replay_buffer), save_dir / "replay_buffer.pt")
            print(f"  saved {ckpt_path} (+ replay_buffer.pt, {len(replay_buffer)} elems)")

    writer.close()


# ============================================================================
# Evaluation
# ============================================================================


def mcts_best_move(game: ConnectFour, network: AlphaZeroNet, num_sims: int, c: float) -> int:
    """Run a search from `game` and return the most-visited (greedy) move.
    Used for evaluation, where we want the agent's strongest deterministic play."""
    root = Node(game)
    mcts(root, network, num_sims, c)
    return max(root.children, key=lambda a: root.children[a].n)


@torch.no_grad()
def play_eval_game(network: AlphaZeroNet, opponent: AlphaZeroNet | None,
                   num_sims: int, c: float, network_plays_x: bool,
                   num_random_opening: int = 0) -> int:
    """Play one game. `opponent=None` means a uniform-random opponent.
    The first `num_random_opening` plies are uniform-random for BOTH sides, so each
    eval game starts from a distinct opening — otherwise deterministic greedy play
    makes every same-colored game identical (a 2-game match in disguise).
    Returns the result from the network's perspective: +1 win, -1 loss, 0 draw."""
    game = ConnectFour()
    network_side = "x" if network_plays_x else "o"
    ply = 0
    while not game.is_terminal():
        if ply < num_random_opening:
            legal = game.legal_moves()
            move = legal[np.random.randint(len(legal))]
        elif game.turn == network_side:
            move = mcts_best_move(game, network, num_sims, c)
        elif opponent is None:
            legal = game.legal_moves()
            move = legal[np.random.randint(len(legal))]
        else:
            move = mcts_best_move(game, opponent, num_sims, c)
        game = game.make_move(move)
        ply += 1
    result = game.result()  # +1 x, -1 o, 0 draw
    return result if network_side == "x" else -result


def evaluate(network: AlphaZeroNet, opponent: AlphaZeroNet | None,
             num_games: int, num_sims: int, c: float,
             num_random_opening: int = 0) -> tuple[float, tuple[int, int, int]]:
    """Play `network` vs `opponent` (None = random), alternating who starts.
    `num_random_opening` random plies diversify openings so the match isn't just
    two deterministic games replayed. Returns (score, (wins, draws, losses))
    where score = (wins + 0.5*draws) / num_games."""
    was_training = network.training
    network.eval()
    if opponent is not None:
        opponent.eval()

    wins = draws = losses = 0
    for i in range(num_games):
        result = play_eval_game(network, opponent, num_sims, c, network_plays_x=(i % 2 == 0),
                                num_random_opening=num_random_opening)
        if result > 0:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1

    if was_training:
        network.train()
    score = (wins + 0.5 * draws) / num_games
    return score, (wins, draws, losses)


# ============================================================================
# Entry point
# ============================================================================


def main() -> None:
    # Usage: python alphazero.py [config.json]   (defaults to ./config.json)
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    config = load_config(config_path)
    train(config)


if __name__ == "__main__":
    main()
