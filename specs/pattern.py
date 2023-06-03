# Result file structure
results = {
    "dataset_name": "dataset_name",
    "dataset_path": "/path/to/dataset/",
    "dataset_hash": "1234567890",
    "dataset_type": "classification",
    "runs": [
        {
            "uid": "short_hash",
            "seed": 1,
            "filtering": {
                "cropping": {
                    "x": 0,
                    "y": 0,
                    "width": 100,
                    "height": 100
                },
                "resampling": {
                    "threshold": 0.5
                }
            },
            "splitting": {
                "indexes_path": "/path/to/indexes/1/",
                "params": {}
            },
            "cross_validation": {
                "indexes_path": "/path/to/indexes/1/",
                "params": {}
            },
            "normalization": {
                "class_type": {},
                "params": {},
                "path": ""
            },
            "augmentation":{
                "pipeline": {},
                "path": ""
            },
            "preprocessing": {
                "pipeline": {},
                "path": ""
            },
            "model": {
                "uid": "short_hash",
                "starting_checkpoint": "/path/to/checkpoint/1/",
                "checkpoint": "/path/to/model/1/", // or ["/path/to/model/1/", "/path/to/model/2/"]
                "name": "CNN",
                "type": "tensorflow", // or ["pytorch", "sklearn"]
            },
            "runtime": "00:00:00",
            "scores": {
                "cv_scores": [

                ],
                "prediction_path": "/path/to/predictions/1/",
                "accuracy": 0.9,
                "precision": 0.9,
                "recall": 0.9,
                "f1": 0.9
            }
        },
        {
            "uid": "short_hash",
            "seed": 1,
            "filtering": {
                "cropping": {
                    "x": 0,
                    "y": 0,
                    "width": 100,
                    "height": 100
                },
                "resampling": {
                    "threshold": 0.5
                }
            },
            "splitting": {
                "indexes_path": "/path/to/indexes/1/",
                "params": {}
            },
            "cross_validation": {
                "indexes_path": "/path/to/indexes/1/",
                "params": {}
            },
            "normalization": {
                "class_type": {},
                "params": {},
                "path": ""
            },
            "augmentation":{
                "pipeline": {},
                "path": ""
            },
            "preprocessing": {
                "pipeline": {},
                "path": ""
            },
            "model": {
                "uid": "short_hash",
                "name": "Stacking",
                "type": "stack",
                "stack_model": {
                    "uid": "short_hash",
                    "starting_checkpoint": "/path/to/checkpoint/1/",
                    "checkpoint": "/path/to/model/1/", // or ["/path/to/model/1/", "/path/to/model/2/"]
                    "name": "Random Forest",
                    "type": "sklearn",
                },
                "stack": [
                    {
                        "uid": "short_hash",
                        "starting_checkpoint": "/path/to/checkpoint/1/",
                        "checkpoint": "/path/to/model/1/", // or ["/path/to/model/1/", "/path/to/model/2/"]
                        "name": "CNN",
                        "type": "tensorflow",
                    },
                    {
                        "uid": "short_hash",
                        "starting_checkpoint": "/path/to/checkpoint/1/",
                        "checkpoint": "/path/to/model/1/", // or ["/path/to/model/1/", "/path/to/model/2/"]
                        "name": "CNN",
                        "type": "tensorflow",
                    },
                    {
                        "uid": "short_hash",
                        "starting_checkpoint": "/path/to/checkpoint/1/",
                        "checkpoint": "/path/to/model/1/", // or ["/path/to/model/1/", "/path/to/model/2/"]
                        "name": "CNN",
                        "type": "pytorch",
                    },
                ],
                
            },
            "runtime": "00:00:00",
            "scores": {
                "cv_scores": [

                ],
                "prediction_path": "/path/to/predictions/1/",
                "accuracy": 0.9,
                "precision": 0.9,
                "recall": 0.9,
                "f1": 0.9
            }
        },
    ]
}


# Pool Config file
training_config = {
    "seed": [42],  # int
    "random_scope": "dataset",  # str - ["dataset", "global", "models"]
    # List[str] - [path1, path2, ...]
    "paths": [("data/test_data", [1,2,3]), "data/_Raisin/Raisin_Tavernier_830_GFratio", "data/_RefSet/ALPINE_C_424_Murguzur_RMSE1.16"],
    # List[(Splitter, Dict)]  - [splitter, params] -> to create dataset indexes trees
    "pre_indexation": {
        "step_1": {
            "type": "filter",
            "method": [("crop", filters.Crop, {"start":100, "end":500}), ("crop", filters.Crop, {"start":0, "end":1000})],# None],
        },
        "step_2": {
            "type": "filter",
            "method": [("resample", filters.Uniform_FT_Resample, {"resample_size": 800})],# None],
        }
    },
    "indexation": [
        ("random_split", indexer.RandomSampling, {"test_size": 0.2}, {}),
        ("random_cv", indexer.RandomSampling, {"folds": 4, "repeat": 1}, {}),
        # None,
        ("random_cv", indexer.SXPY, {"folds": 4, "repeat": 1}, {'metric':"euclidean", 'pca_components':250}),
    ],
    "post_indexation": {
        "step_1": {
            "type": "augmentation",
            "method": [
                # None,
                [(6, augmentation.Rotate_Translate())],
                [(3, augmentation.Rotate_Translate()),(2, augmentation.Random_X_Operation()),(1, augmentation.Random_Spline_Addition()),],
                [(3, augmentation.Rotate_Translate()),(2, augmentation.Random_X_Operation()),(2, augmentation.Random_Spline_Addition()),]
            ],
        },
        "step_2": {
            "type": "preprocessing",
            "method": [
                # None,
                preprocessings.id_preprocessing(),
                [("id", pp.IdentityTransformer()), ('haar', pp.Haar()), ('savgol', pp.SavitzkyGolay())],
                preprocessings.decon_set(),
            ]
        },
    },
    "models": [
        # ()
        (regressors.Transformer_NIRS, {'batch_size':500, 'epoch':10000, 'verbose':0, 'patience':1000, 'optimizer':'adam', 'loss':'mse'}),
        (XGBRegressor, {"n_estimators":200, "max_depth":50}),
        (PLSRegression, {"n_components":50}),
        (regressors.Decon_SepPo, {'batch_size':50, 'epoch':10000, 'verbose':0, 'patience':1000, 'optimizer':'adam', 'loss':'mse'}),
    ],
}




# FORMATS of PROCESSING transformerMixin
# (inst|class|function)
# (class|function, params)
# ("name", inst|class|function)
# ("name", class|function, params)

# FORMATS of AUGMENTATION transformerMixin
# (inst|class|function)
# (inst|class|function, nb)
# (class|function, params)
# (class|function, params, nb)
# ("name", inst|class|function)
# ("name", inst|class|function, nb)
# ("name", class|function, params)
# ("name", class|function, params, nb)

# FORMATS of MODELS estimators
# (inst|class|function)
# (inst|class|function, params)
# ("name", inst|class|function)
# ("name", inst|class|function, params)