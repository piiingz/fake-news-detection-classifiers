import warnings

import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from classifiers.evaluation_metrics import auc, acc, f1
from handle_datasets.paths import XGB_PATH, XGB_DEFAULT_PATH

warnings.filterwarnings('ignore')


def tune_parameters(X_train, y_train):
    # Parameter Tuning
    # Best params so far: {'gamma': 1e-09, 'learning_rate': 0.16, 'max_depth': 10, 'min_child_weight': 1, 'n_estimators': 500})
    model = xgb.XGBClassifier(silent=True)

    # param_dist = {"max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    #             }

    param_dist = {"max_depth": [5],
                "learning_rate": [0.3, 0.1, 0.01],
                "colsample_bytree": [0.3, 0.8],
                "subsample": [0.8, 1],
                "gamma": [0, 1, 5]
                }

    grid_search = GridSearchCV(model, param_grid=param_dist, cv=5, verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def train_xgb(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train, y_val, y_test, is_default: bool = False):
    # specify parameters via map
    # 
    
    """

    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=9,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, silent=True,
              subsample=1, tree_method='exact', validate_parameters=1,
              verbosity=None)

    Supervised:
    (XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=1, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, silent=True,
              subsample=0.8, tree_method='exact', validate_parameters=1,
              verbosity=None),
               {'colsample_bytree': 0.3, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8})

    Weak:
    (XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.3, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, silent=True,
              subsample=1, tree_method='exact', validate_parameters=1,
              verbosity=None),
              {'colsample_bytree': 0.3, 'gamma': 0, 'learning_rate': 0.3, 'max_depth': 5, 'subsample': 1})

    """

    if is_default:
        xgb_model = xgb.XGBClassifier()
    else:
        xgb_model = xgb.XGBClassifier(
            gamma=0,
            learning_rate=0.3,
            max_depth=5,
            n_estimators=100,
            subsample=1,
            colsample_bytree=0.3,
            verbosity=None
            )

    eval_set = [(X_val, y_val)]
    xgb_model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, early_stopping_rounds=10, verbose=True)

    """    # feature importance
        print(xgb_model.feature_importances_)
        # plot
        pyplot.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)

        pyplot.savefig('feature_importance.png')
    """
    from matplotlib import pyplot
    from matplotlib.pylab import rcParams
    rcParams.update({'font.size': 22})
    rcParams['figure.figsize'] = 30, 20

    xgb.plot_importance(xgb_model, max_num_features=20)
    pyplot.show()

    if is_default:
        xgb_model.save_model(XGB_DEFAULT_PATH)
        print(xgb)
        pyplot.savefig('feature_importance_default.png')
    else:
        xgb_model.save_model(XGB_PATH)
        pyplot.savefig('feature_importance.png')

    print("AUC score: ", auc(xgb_model, X_val, X_test, y_val, y_test))
    print("F1 score: ", f1(xgb_model, X_val, X_test, y_val, y_test))
    print("Accuracy: ", acc(xgb_model, X_val, X_test, y_val, y_test))
    print("Confusion matrix validation: \n" + str(confusion_matrix(y_val, xgb_model.predict(X_val))))
    print("Confusion matrix test: \n" + str(confusion_matrix(y_test, xgb_model.predict(X_test))))
