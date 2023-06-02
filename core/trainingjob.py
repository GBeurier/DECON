# from typing import Dict, List, Union


# # a class that defines a run of training based of specs/pattern.json
class TrainingJob:
    def __init__(self, dataset: str, data_config, model_config) -> None:
        self.dataset = dataset
        self.data_config = data_config
        self.model_config = model_config

    # def train(self) -> None:
    #     pass

    # def stack_existing(self) -> None:
    #     pass

    # def resume_training(self, model_name: str = "", run_id: str = "") -> None:
    #     pass

    # def tune_model(self, tuning_config: Dict) -> None:
    #     pass

    # def predict(self, predict_set: str = "", model_name: str = "", run_id: str = "") -> None:
    #     pass