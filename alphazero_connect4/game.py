from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

ROWS: int = 6
COLS: int = 7
CONNECT: int = 4


class ConnectFour:

    def __init__(self, state: npt.NDArray[np.float64] | None = None, turn: str = "x") -> None:
        self.state: npt.NDArray[np.float64] = np.zeros((ROWS, COLS)) if state is None else state
        self.turn: str = turn

    def reset(self) -> None:
        self.state = np.zeros((ROWS, COLS))

    def copy(self) -> ConnectFour:
        return ConnectFour(self.state.copy(), self.turn)

    def legal_moves(self) -> list[int]:
        return [c for c in range(COLS) if self.state[0, c] == 0]

    def make_move(self, action: int) -> ConnectFour:
        state = self.state.copy()
        for row in range(ROWS - 1, -1, -1):
            if state[row, action] == 0:
                state[row, action] = 1 if self.turn == "x" else -1
                break
        turn = "o" if self.turn == "x" else "x"
        return ConnectFour(state, turn)

    def is_terminal(self) -> bool:
        if self.result() != 0:
            return True
        return len(self.legal_moves()) == 0

    def result(self) -> int:
        # +1 for x, -1 for o, 0 for draw / ongoing
        for r in range(ROWS):
            for c in range(COLS - CONNECT + 1):
                window = self.state[r, c:c + CONNECT]
                s = int(np.sum(window))
                if abs(s) == CONNECT:
                    return int(np.sign(s))
        for r in range(ROWS - CONNECT + 1):
            for c in range(COLS):
                window = self.state[r:r + CONNECT, c]
                s = int(np.sum(window))
                if abs(s) == CONNECT:
                    return int(np.sign(s))
        for r in range(ROWS - CONNECT + 1):
            for c in range(COLS - CONNECT + 1):
                window = np.array([self.state[r + i, c + i] for i in range(CONNECT)])
                s = int(np.sum(window))
                if abs(s) == CONNECT:
                    return int(np.sign(s))
        for r in range(CONNECT - 1, ROWS):
            for c in range(COLS - CONNECT + 1):
                window = np.array([self.state[r - i, c + i] for i in range(CONNECT)])
                s = int(np.sum(window))
                if abs(s) == CONNECT:
                    return int(np.sign(s))
        return 0

    def render(self) -> None:
        symbols: dict[int, str] = {0: ".", 1: "X", -1: "O"}
        print(" ".join(str(c) for c in range(COLS)))
        for r in range(ROWS):
            print(" ".join(symbols[int(self.state[r, c])] for c in range(COLS)))
        print()

    def to_tensor(self) -> torch.Tensor:
        if self.turn == "x":
            current = (self.state == 1).astype(np.float32)
            opponent = (self.state == -1).astype(np.float32)
        else:
            current = (self.state == -1).astype(np.float32)
            opponent = (self.state == 1).astype(np.float32)
        return torch.tensor(np.stack([current, opponent], axis=0))

    def state_key(self) -> bytes:
        return self.state.tobytes()
