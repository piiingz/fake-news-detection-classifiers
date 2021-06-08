from handle_datasets.load_datasets import load_all_weak_supervised_tf_idf_train_val_test, \
    load_all_supervised_tf_idf_train_val_test
from classifiers.xgboost.xgboost_model import *

is_weak = 1  # 1 for weak, 0 for supervised

if __name__ == '__main__':
    if is_weak:
        print("Weak")
        train, val, test = load_all_weak_supervised_tf_idf_train_val_test()

    else:
        print("Supervised")
        train, val, test = load_all_supervised_tf_idf_train_val_test()

    X_train = train.drop(['label'], axis=1)
    y_train = train['label']

    print("Best params: ")
    print(tune_parameters(X_train, y_train))
