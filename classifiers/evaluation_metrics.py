from sklearn import metrics


def auc(m, train, test, y_train, y_test):
    return (metrics.roc_auc_score(y_train, m.predict_proba(train)[:, 1]),
            metrics.roc_auc_score(y_test, m.predict_proba(test)[:, 1]))


def f1(m, train, test, y_train, y_test):
    return (metrics.f1_score(y_train, m.predict(train)),
            metrics.f1_score(y_test, m.predict(test)))


def acc(m, train, test, y_train, y_test):
    return (metrics.accuracy_score(y_train, m.predict(train)),
            metrics.accuracy_score(y_test, m.predict(test)))
