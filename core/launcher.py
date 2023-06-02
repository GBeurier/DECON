from itertools import product

import datacache
import pipeliner


def generate_combinations(steps):
    methods = []
    for step in steps.values():
        methods.append(step['methods'])
    print(methods)
    return list(product(*methods))


# TRAINING
training_config = {
    "seed": [42],  # int
    "random_scope": "dataset",  # str - ["dataset", "global", "models"]
    "paths": ["path1", "path2"],  # List[str] - [path1, path2, ...] > 3 formats: "path" (get XCal XVal yCal yVal .csv or tgz), (path, path) (get XCal XVal yCal yVal .csv or tgz), "path:dataset:split" (get XCal XVal yCal yVal .csv or tgz)
    "indexation": [],  # List[(Splitter, Dict)]  - [splitter, params] -> to create dataset indexes trees
    "pre_indexation": {
        "step_1": {
            "type": "filter",
            "methods": [],  # List[List[(TransformerMixin, Dict)]]
        },
        "step_2": {
            "type": "augmentation",
            "methods": [],  # List[List[(TransformerMixin, Dict)]]
        },
    },
    "post_indexation": {
        "step_1": {
            "type": "filter",
            "method": [],  # List[List[(TransformerMixin, Dict)]]
        },
        "step_2": {
            "type": "preprocessing",
            "method": [],  # List[List[(TransformerMixin, Dict)]]
        },
    },
    "models": [
        [],  # List[(Estimator, Dict)]  - [estimator, params] -> to create sklearn pipeline
    ],
}

# data_configs = {}
# for dataset in training_config["data"]["paths"]:
#     # 1. Load data
#     # init cache
#     pass

pre_indexation_steps = generate_combinations(training_config["pre_indexation"])
post_indexation_steps = generate_combinations(training_config["post_indexation"])





    # 2. Pre-indexation processing
    # 3. Indexation
    # 4. Cross validating data
    # 5. Preprocessing data
    

    
    
    # 3. Splitting data
    # 4. Cross validating data
    # 5. Preprocessing data
    # 6. Training models
    # 7. Evaluating models
    # 8. Stacking models
    # 9. Tuning models
    # 10. Predicting models
    pass