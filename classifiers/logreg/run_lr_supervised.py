from config import SUPERVISED_BEST_C
from handle_datasets.load_datasets import load_all_supervised_tf_idf_train_val_test
from classifiers.logreg.lr_model import train_lr, print_lr_params

if __name__ == '__main__':
    print("Starting..")
    print_lr_params()
    print("C value: ", SUPERVISED_BEST_C)

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

    train_lr(X_train, X_val, X_test, y_train, y_val, y_test, SUPERVISED_BEST_C)
