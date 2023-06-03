from typing import Union, List, Dict


def train_pool(config: Dict) -> List[Dict]:
    pass


def train(dataset: str, run_config: Union[Dict, List]) -> None:
    pass


def resume_training(dataset: str, run_config: Dict, model_name: str = "", run_id: str = "") -> None:
    pass


def stack_existing(dataset: str, run_config: Union[Dict, List]) -> None:
    pass


def tune_model(dataset: str, run_config: Dict, tuning_config: Dict) -> None:
    pass


def predict(dataset: str, predict_set: str = "", model_name: str = "", run_id: str = "") -> None:
    pass
