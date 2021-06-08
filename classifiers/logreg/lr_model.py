from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

from config import MAX_ITER, SOLVER, PENALTY, RANDOM_SEED, CV, TITLE_MAX_FEATURES, CONTENT_MAX_FEATURES


def tune_lr_parameters(X_train, X_val, X_test, y_train, y_val, y_test):
    grid_values = {'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25]}

    clf = LogisticRegression(max_iter=MAX_ITER, solver=SOLVER, penalty=PENALTY, random_state=RANDOM_SEED)
    grid_clf = GridSearchCV(clf, param_grid=grid_values, cv=CV)
    grid_clf.fit(X_train, y_train)

    print("Best params: ", grid_clf.best_params_)
    print("Acc: ", grid_clf.best_score_)

    print("On validation")
    y_pred = grid_clf.predict(X_val)
    print_scores(y_val, y_pred)

    print("On test")
    y_test_pred = grid_clf.predict(X_test)
    print_scores(y_test, y_test_pred)

    return grid_clf.best_estimator_, grid_clf.best_params_, grid_clf.best_score_


def train_lr(X_train, X_val, X_test, y_train, y_val, y_test, best_C):

    clf = LogisticRegression(max_iter=MAX_ITER, solver=SOLVER, penalty=PENALTY, random_state=RANDOM_SEED, C=best_C)
    clf.fit(X_train, y_train)

    print("On train")
    y_train_pred = clf.predict(X_train)
    print_scores(y_train, y_train_pred)

    print("On validation")
    y_val_pred = clf.predict(X_val)
    print_scores(y_val, y_val_pred)

    print("On test")
    y_test_pred = clf.predict(X_test)
    print_scores(y_test, y_test_pred)


def print_lr_params():
    print("Title max features: ", TITLE_MAX_FEATURES)
    print("Content max features: ", CONTENT_MAX_FEATURES)
    print("Max iterations: ", MAX_ITER)
    print("Solver: ", SOLVER)
    print("Penalty: ", PENALTY)


def print_scores(y_true, y_pred):
    print('Accuracy Score : ' + str(accuracy_score(y_true, y_pred)))
    print('Precision Score : ' + str(precision_score(y_true, y_pred)))
    print('Recall Score : ' + str(recall_score(y_true, y_pred)))
    print('F1 Score : ' + str(f1_score(y_true, y_pred)))
    print("Confusion matrix: \n" + str(confusion_matrix(y_true, y_pred)))
