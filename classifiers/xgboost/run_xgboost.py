from handle_datasets.load_datasets import load_all_weak_supervised_tf_idf_train_val_test, \
    load_all_supervised_tf_idf_train_val_test
from classifiers.xgboost.xgboost_model import *
import sys

is_weak = 0     # 1 for weak, 0 for supervised

if __name__ == '__main__':
    if is_weak:
        print("Weak")
        train, val, test = load_all_weak_supervised_tf_idf_train_val_test()

    else:
        print("Supervised")
        train, val, test = load_all_supervised_tf_idf_train_val_test()

    X_train = train.drop(['label'], axis=1)
    y_train = train['label']

    X_val = val.drop(['label'], axis=1)
    y_val = val['label']

    X_test = test.drop(['label'], axis=1)
    y_test = test['label']

    # tune_parameters(X_train, y_train)
    is_default = bool(int(sys.argv[1]))     # 0 or 1
    train_xgb(X_train, X_val, X_test, y_train, y_val, y_test, is_default)
