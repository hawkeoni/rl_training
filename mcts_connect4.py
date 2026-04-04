import numpy as np

ROWS = 6
COLS = 7
CONNECT = 4


class ConnectFour:

    def __init__(self, state=None, turn="x"):
        self.state = np.zeros((ROWS, COLS))
        self.turn = turn
        if state is not None:
            self.state = state

    def reset(self):
        self.state = np.zeros((ROWS, COLS))

    def copy(self):
        return ConnectFour(self.state.copy(), self.turn)

    def legal_moves(self):
        # A column is legal if the top row is empty
        return [c for c in range(COLS) if self.state[0, c] == 0]

    def make_move(self, action):
        state = self.state.copy()
        # Drop piece into the lowest empty row of the column
        for row in range(ROWS - 1, -1, -1):
            if state[row, action] == 0:
                state[row, action] = 1 if self.turn == "x" else -1
                break
        turn = "o" if self.turn == "x" else "x"
        return ConnectFour(state, turn)

    def is_terminal(self):
        if self.result() != 0:
            return True
        return len(self.legal_moves()) == 0

    def result(self):
        # +1 for x, -1 for o, 0 for draw / ongoing
        # Check horizontal
        for r in range(ROWS):
            for c in range(COLS - CONNECT + 1):
                window = self.state[r, c:c + CONNECT]
                s = int(np.sum(window))
                if abs(s) == CONNECT:
                    return int(np.sign(s))
        # Check vertical
        for r in range(ROWS - CONNECT + 1):
            for c in range(COLS):
                window = self.state[r:r + CONNECT, c]
                s = int(np.sum(window))
                if abs(s) == CONNECT:
                    return int(np.sign(s))
        # Check diagonal (top-left to bottom-right)
        for r in range(ROWS - CONNECT + 1):
            for c in range(COLS - CONNECT + 1):
                window = np.array([self.state[r + i, c + i] for i in range(CONNECT)])
                s = int(np.sum(window))
                if abs(s) == CONNECT:
                    return int(np.sign(s))
        # Check anti-diagonal (bottom-left to top-right)
        for r in range(CONNECT - 1, ROWS):
            for c in range(COLS - CONNECT + 1):
                window = np.array([self.state[r - i, c + i] for i in range(CONNECT)])
                s = int(np.sum(window))
                if abs(s) == CONNECT:
                    return int(np.sign(s))
        return 0

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        print(" ".join(str(c) for c in range(COLS)))
        for r in range(ROWS):
            print(" ".join(symbols[int(self.state[r, c])] for c in range(COLS)))
        print()

    def state_key(self):
        return self.state.tobytes()



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


def main():
    game = ConnectFour()
    mcts = MCTS(c=2.0)
    human_side = input("Play as x or o? ").strip().lower()
    assert human_side in ("x", "o")

    while not game.is_terminal():
        game.render()
        if game.turn == human_side:
            legal = game.legal_moves()
            print(f"Legal columns (0-{COLS - 1}): {legal}")
            move = int(input("Your move: "))
            assert move in legal
        else:
            mcts.train(num_iters=10000)
            best = mcts.best_child()
            move = best.action
            print(f"MCTS plays {move}  (visits: {best.n}, value: {best.t / best.n:.3f})")

        game = game.make_move(move)
        # Reuse subtree if the child was previously explored
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


if __name__ == "__main__":
    main()