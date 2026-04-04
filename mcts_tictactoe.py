import numpy as np

class TicTacToe:

    def __init__(self, state=None, turn="x"):
        self.state = np.zeros((3, 3))
        self.turn = turn
        if state is not None:
            self.state = state
    
    def reset(self):
        self.state = np.zeros((3, 3))

    def copy(self):
        return TicTacToe(self.state.copy(), self.turn)

    def legal_moves(self):
        return (self.state.flatten() == 0).nonzero()[0].tolist()

    def make_move(self, action):
        state = self.state.copy()
        i = action // 3
        j = action % 3
        state[i, j] = 1 if self.turn == "x" else -1
        turn = "o" if self.turn == "x" else "x"
        return TicTacToe(state, turn)

    def is_terminal(self):
        if self.result() != 0:
            return True
        return len(self.legal_moves()) == 0

    def result(self):
        # +1 for crosses
        # -1 for circles
        # 0 for draw
        for i in range(3):
            row_sum = np.sum(self.state[i, :])
            col_sum = np.sum(self.state[:, i])
            if abs(row_sum) == 3:
                return int(np.sign(row_sum))
            if abs(col_sum) == 3:
                return int(np.sign(col_sum))
        diag1 = np.sum(np.diag(self.state))
        diag2 = np.sum(np.diag(np.fliplr(self.state)))
        if abs(diag1) == 3:
            return int(np.sign(diag1))
        if abs(diag2) == 3:
            return int(np.sign(diag2))
        return 0
    
    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        for i in range(3):
            print(" ".join(symbols[int(self.state[i, j])] for j in range(3)))
        print()

    def state_key(self):
        return self.state.tobytes()



class Node:

    def __init__(self, game: TicTacToe, parent: "Node" = None, action: int = None):
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
        self.root = root if root is not None else Node(TicTacToe())
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
    game = TicTacToe()
    mcts = MCTS(c=2.0)
    human_side = input("Play as x or o? ").strip().lower()
    assert human_side in ("x", "o")

    while not game.is_terminal():
        game.render()
        if game.turn == human_side:
            legal = game.legal_moves()
            print(f"Legal moves (0-8): {legal}")
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