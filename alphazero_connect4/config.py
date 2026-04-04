from pydantic import BaseModel


class TrainConfig(BaseModel):
    save_folder: str = "connect4_alphazero_checkpoints"
    num_epochs: int = 10
    num_games_per_epoch: int = 100
    num_sub_epochs: int = 5
    num_mcts_sims: int = 800
    learning_rate: float = 1e-3
    batch_size: int = 64
    c: float = 1.5
    eval_games: int = 20
    eval_mcts_sims: int = 200


class PlayConfig(BaseModel):
    model_path: str
    num_mcts_sims: int = 800
