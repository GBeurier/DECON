
def loop_data(): # Raw loading then filters (resampling, cropping, discretization, quantization, etc.)
    for path in dataset_list:
        # load raw data
        X, y, X_valid, y_valid = load_data(path, resampling, resample_size)
        # apply filters on X
        X, y, X_valid, y_valid
    


def benchmark_dataset(dataset_list, split_configs, cv_configs, augmentations, preprocessings, models, SEED, *, bins=None, resampling=None, resample_size=0):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    classification_mode = False if bins is None else True
    name_classif = "" if classification_mode is False else '_Cl' + str(bins) + '_'

    for path in dataset_list:
        # Infos
        dataset_name, results, results_file = init_log(path)
        target_RMSE, best_current_model = "-1", "None"
        if len(results.keys()) > 0:
            best_current_model = list(results.keys())[0]
            target_RMSE = results[best_current_model]["RMSE"]

        folder = os.path.join("results", dataset_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        desc = (dataset_name, path, results_file, results, SEED)

        # Load data
        print("="*10, str(dataset_name).upper(), end=" ")
        X, y, X_valid, y_valid = load_data(path, resampling, resample_size)
        print("="*10, X.shape, y.shape, X_valid.shape, y_valid.shape)

        # Split data
        for split_config in split_configs:
            # print("Split >", split_config)
            X_train, y_train, X_test, y_test = X, y, X_valid, y_valid
            discretizer = None
            if classification_mode:
                discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='uniform')
                discretizer.fit(y_train)
                y_train = discretizer.transform(y_train.reshape((-1, 1)))
                y_test = discretizer.transform(y_test.reshape((-1, 1)))
                print("Discretized shape", y_train.shape, y_test.shape)
            # else:
            #     y_train = y_train.reshape((-1, 1))
            #     y_test = y_test.reshape((-1, 1))

            name_split = "NoSpl"
            if split_config is not None:
                train_index, test_index = model_selection.train_test_split_idx(X, y=y, **split_config)
                X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]

                # Replace validation set by test set if validation set is empty
                if X_valid.size == 0:
                    X_valid, y_valid = X_test, y_test

                name_split = 'Spl_' + str(split_config['method']) + "_" + str(hash(frozenset(split_config)))[0:3]

            # print("Split >", X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_valid.shape, y_valid.shape)
            
            # Generate training sets
            for cv_config in cv_configs:
                # print("CV >", cv_config)

                folds = iter([([],[])])
                folds_size = 1
                name_cv = "NoCV"
                time_str_cv = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
                if cv_config is not None:
                    cv = RepeatedKFold(random_state=SEED, **cv_config)
                    folds = cv.split(X_train)
                    # cv = regressor_stratified_cv(**cv_config, group_count=5, random_state=SEED, strategy='uniform')
                    # folds = cv.split(X_train, y_train)
                    folds_size = cv.get_n_splits()
                    name_cv = "CV_" + str(cv_config['n_splits']) + "_" + str(cv_config['n_repeats'])
                
                # Loop data
                fold_i = 1
                cv_predictions = {}
                for train_index, test_index in folds:
                    if len(train_index) > 0:
                        X_tr, y_tr, X_te, y_te = X_train[train_index], y_train[train_index], X_train[test_index], y_train[test_index]
                    else:
                        X_tr, y_tr, X_te, y_te = X_train, y_train, X_test, y_test

                    name_fold = "Fold_" + str(fold_i) + '(' + str(folds_size) + ')'
                    # print(name_fold, X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)
                    fold_i += 1
                    
                    # Loop augmentations
                    for augmentation_config in augmentations:
                        name_augmentation, augmentation_pipeline = get_augmentation(augmentation_config)
                        if augmentation_pipeline is not None:
                            X_tr, y_tr = augmentation_pipeline.transform(X_tr, y_tr)
                        # print(name_augmentation, X_tr.shape, y_tr.shape)
                        data = (X_tr, y_tr, X_valid, y_valid)

                        # Loop preprocessing
                        for preprocessing in preprocessings:
                            name_preprocessing = 'NoPP_'
                            if preprocessing is not None:
                                name_preprocessing = 'PP_' + str(len(preprocessing)) + "_" + str(hash(frozenset(preprocessing)))[0:5]

                            for model in models:
                                name_model = model[0].name()
                                isThreadable = False
                                if isinstance(model[0], regressors.NN_NIRS_Regressor):
                                    # print(name_model, "is Neural Net")
                                    X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(
                                        preprocessing, X_tr, y_tr, X_te, y_te, type="augmentation", classification_mode=classification_mode
                                    )
                                else:
                                    if classification_mode:
                                        print('Error:', "Can only use classification on neural nets models")
                                        return

                                    # print(name_model, "is Machine Learning")
                                    X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(
                                        preprocessing, X_tr, y_tr, X_te, y_te, type="union"
                                    )
                                    isThreadable = True

                                time_str = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
                                run_name = "-".join([name_model, name_classif, name_split, name_cv, name_fold, name_augmentation, name_preprocessing, str(SEED), time_str])
                                run_key = "-".join([name_model, name_classif, name_split, name_cv, name_augmentation, name_preprocessing, str(SEED), time_str_cv])

                                print(run_name, X_tr.shape, y_tr.shape, X_test_pp.shape, y_test_pp.shape)

                                # print(">"*10, X_tr[0], y_test_pp[0])
                                cb_predict = get_callback_predict(dataset_name, name_model, path, SEED, target_RMSE, best_current_model, discretizer)
                                
                                regressor = model[0].model(X_tr, y_tr, X_test_pp, y_test_pp, run_name=run_name, cb=cb_predict, params=model[1], desc=desc, discretizer=discretizer)
                                transformers = (y_scaler, transformer_pipeline, regressor)

                                if isThreadable:
                                    y_pred, datasheet = evaluate_pipeline(desc, run_name, data, transformers, target_RMSE, best_current_model, discretizer) # TODO thread this
                                else:
                                    y_pred, datasheet = evaluate_pipeline(desc, run_name, data, transformers, target_RMSE, best_current_model, discretizer)

                                if name_cv != "NoCV":
                                    if run_key not in cv_predictions:
                                        cv_predictions[run_key] = []
                                    cv_predictions[run_key].append({'pred': y_pred, 'RMSE': float(datasheet['RMSE'])})
                
                if name_cv != "NoCV":
                    for key, val in cv_predictions.items():
                        RMSE_TOT = sum(item['RMSE'] for item in val)
                        factor = 1. / (len(val) - 1)
                        weights = [factor * (RMSE_TOT - item['RMSE']) / RMSE_TOT for item in val]
                        y_pred = np.sum([weights[i]*val[i]['pred'] for i in range(len(val))], axis=0)
                        datasheet = get_datasheet(dataset_name, key, path, SEED, y_valid, y_pred)
                        results[key + "_CV"] = datasheet

                results = OrderedDict(sorted(results.items(), key=lambda k_v: float(k_v[1]["RMSE"])))
                with open(results_file, "w") as fp:
                    json.dump(results, fp, indent=4)



def benchmark_dataset_multiple(dataset_list, split_configs, cv_configs, augmentations, preprocessings, models, SEED, *, bins=None, resampling=None, resample_size=0):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    classification_mode = False if bins is None else True
    name_classif = "" if classification_mode is False else '_Cl' + str(bins) + '_'

    for path in dataset_list:
        # Infos
        dataset_name, results, results_file = init_log(path)
        target_RMSE, best_current_model = "-1", "None"
        if len(results.keys()) > 0:
            best_current_model = list(results.keys())[0]
            target_RMSE = results[best_current_model]["RMSE"]

        folder = os.path.join("results", dataset_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        desc = (dataset_name, path, results_file, results, SEED)

        # Load data
        print("="*10, str(dataset_name).upper(), end=" ")
        X, y, X_valid, y_valid = load_data_multiple(path, resampling, resample_size)
        print("="*10, X.shape, y.shape, X_valid.shape, y_valid.shape)

        # Split data
        for split_config in split_configs:
            # print("Split >", split_config)
            X_train, y_train, X_test, y_test = X, y, X_valid, y_valid
            discretizer = None
            if classification_mode:
                discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='uniform')
                discretizer.fit(y_train)
                y_train = discretizer.transform(y_train.reshape((-1, 1)))
                y_test = discretizer.transform(y_test.reshape((-1, 1)))
                print("Discretized shape", y_train.shape, y_test.shape)
            # else:
            #     y_train = y_train.reshape((-1, 1))
            #     y_test = y_test.reshape((-1, 1))

            name_split = "NoSpl"
            if split_config is not None:
                train_index, test_index = model_selection.train_test_split_idx(X, y=y, **split_config)

                X_train, y_train, X_test, y_test = [sub_x[train_index] for sub_x in X], y[train_index], [sub_x[test_index] for sub_x in X], y[test_index]

                # Replace validation set by test set if validation set is empty
                if X_valid.size == 0:
                    X_valid, y_valid = X_test, y_test

                name_split = 'Spl_' + str(split_config['method']) + "_" + str(hash(frozenset(split_config)))[0:3]

            # print("Split >", X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_valid.shape, y_valid.shape)
            
            # Generate training sets
            for cv_config in cv_configs:
                # print("CV >", cv_config)

                folds = iter([([],[])])
                folds_size = 1
                name_cv = "NoCV"
                time_str_cv = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
                if cv_config is not None:
                    cv = RepeatedKFold(random_state=SEED, **cv_config)
                    folds = cv.split(X_train[0])
                    # cv = regressor_stratified_cv(**cv_config, group_count=5, random_state=SEED, strategy='uniform')
                    # folds = cv.split(X_train, y_train)
                    folds_size = cv.get_n_splits()
                    name_cv = "CV_" + str(cv_config['n_splits']) + "_" + str(cv_config['n_repeats'])
                
                # Loop data
                fold_i = 1
                cv_predictions = {}
                for train_index, test_index in folds:
                    if len(train_index) > 0:
                        X_tr, y_tr, X_te, y_te = np.array([sub_x[train_index] for sub_x in X_train]), y_train[train_index], np.array([sub_x[test_index] for sub_x in X_train]), y_train[test_index]
                    else:
                        X_tr, y_tr, X_te, y_te = X_train, y_train, X_test, y_test

                    name_fold = "Fold_" + str(fold_i) + '(' + str(folds_size) + ')'
                    # print(name_fold, X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)
                    fold_i += 1
                    
                    # Loop augmentations
                    for augmentation_config in augmentations:
                        name_augmentation, augmentation_pipeline = get_augmentation(augmentation_config)
                        if augmentation_pipeline is not None:
                            for i in range(len(X_tr)):
                                X_tr[i], y_tr = augmentation_pipeline.transform(X_tr[i], y_tr)
                        # print(name_augmentation, X_tr.shape, y_tr.shape)
                        data = (X_tr, y_tr, X_valid, y_valid)

                        # Loop preprocessing
                        for preprocessing in preprocessings:
                            name_preprocessing = 'NoPP_'
                            if preprocessing is not None:
                                name_preprocessing = 'PP_' + str(len(preprocessing)) + "_" + str(hash(frozenset(preprocessing)))[0:5]

                            for model in models:
                                name_model = model[0].name()
                                if isinstance(model[0], regressors.NN_NIRS_Regressor_Multiple):
                                    print(name_model, "is Neural Net")
                                    print("before transform", X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)
                                    X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data_multiple(
                                        preprocessing, X_tr, y_tr, X_te, y_te, type="augmentation", classification_mode=classification_mode
                                    )
                                    print("after transform", X_test_pp.shape, y_test_pp.shape)
                                else:
                                    if classification_mode:
                                        print('Error:', "Can only use classification on neural nets models")
                                        return

                                    # print(name_model, "is Machine Learning")
                                    X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data_multiple(
                                        preprocessing, X_tr, y_tr, X_te, y_te, type="union"
                                    )

                                time_str = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
                                run_name = "-".join([name_model, name_classif, name_split, name_cv, name_fold, name_augmentation, name_preprocessing, str(SEED), time_str])
                                run_key = "-".join([name_model, name_classif, name_split, name_cv, name_augmentation, name_preprocessing, str(SEED), time_str_cv])

                                print(">>", run_name, X_tr.shape, y_tr.shape, X_test_pp.shape, y_test_pp.shape)

                                # print(">"*10, X_tr[0], y_test_pp[0])
                                data = (X_tr, y_tr, X_test_pp, y_test_pp)
                                cb_predict = get_callback_predict_multiple(dataset_name, name_model, path, SEED, target_RMSE, best_current_model, discretizer)
                                regressor = model[0].model(X_tr, y_tr, X_test_pp, y_test_pp, run_name=run_name, cb=cb_predict, params=model[1], desc=desc, discretizer=discretizer)
                                transformers = (y_scaler, transformer_pipeline, regressor)

                                y_pred, datasheet = evaluate_pipeline_multiple(desc, run_name, data, transformers, target_RMSE, best_current_model, discretizer)
                                if name_cv != "NoCV":
                                    if run_key not in cv_predictions:
                                        cv_predictions[run_key] = []
                                    cv_predictions[run_key].append({'pred': y_pred, 'RMSE': float(datasheet['RMSE'])})
                

                ## BUG HERE Folds and validation are mixed
                # if name_cv != "NoCV":
                #     for key, val in cv_predictions.items():
                #         RMSE_TOT = sum(item['RMSE'] for item in val)
                #         factor = 1. / (len(val) - 1)
                #         weights = [factor * (RMSE_TOT - item['RMSE']) / RMSE_TOT for item in val]
                #         y_pred = np.sum([weights[i]*val[i]['pred'] for i in range(len(val))], axis=0)
                #         datasheet = get_datasheet(dataset_name, key, path, SEED, y_valid, y_pred)
                #         results[key + "_CV"] = datasheet

                results = OrderedDict(sorted(results.items(), key=lambda k_v: float(k_v[1]["RMSE"])))
                with open(results_file, "w") as fp:
                    json.dump(results, fp, indent=4)