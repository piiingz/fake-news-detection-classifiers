from handle_datasets.load_datasets import load_all_supervised_tf_idf_train_val_test

from classifiers.logreg.lr_model import tune_lr_parameters, print_lr_params

if __name__ == '__main__':
    print("Starting..")
    print_lr_params()

    train, val, test = load_all_supervised_tf_idf_train_val_test()
    print("Dataset loaded..")

    print("Train shape: ", train.shape)
    print("Validation shape: ", val.shape)
    print("Test shape: ", test.shape)

    X_train = train.drop(['label'], axis=1)
    y_train = train['label']

    X_val = val.drop(['label'], axis=1)
    y_val = val['label']

    X_test = test.drop(['label'], axis=1)
    y_test = test['label']

    best_est, best_params, best_score = tune_lr_parameters(X_train, X_val, X_test, y_train, y_val, y_test)

    print("Best estimator: ", best_est)
    print("Best params: ", best_params)
    print("Best score: ", best_score)
