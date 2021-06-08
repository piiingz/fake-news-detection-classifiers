from config import IDUN

if IDUN:
    DATA_PATH = 'data/'
else:
    DATA_PATH = '../data/'

NO6_NUMERICAL = DATA_PATH + 'no6_numerical/numerical'

""" Help dictionaries """
DATASET_PATHS = {6: NO6_NUMERICAL}


""" Numerical file names """
TRAIN_BALANCED = '_train_balanced'
TEST_BALANCED = '_test_balanced'

TRAIN_UNBALANCED = '_train_unbalanced'
TEST_UNBALANCED = '_test_unbalanced'

TRAIN_BALANCED_LEMMA = '_train_balanced_lemma'
TEST_BALANCED_LEMMA = '_test_balanced_lemma'

TRAIN_BALANCED_RAW = '_train_balanced_raw'
TEST_BALANCED_RAW = '_test_balanced_raw'

DEVELOPMENT_DATA_PATH = DATA_PATH + 'numerical_test_balanced.pkl'

""" Testset """
TEST_SET_PATH = DATA_PATH + "testset/"
TEST_SET_PATH_REAL = TEST_SET_PATH + "real_cleaned.csv"
TEST_SET_PATH_FAKE = TEST_SET_PATH + "fake_cleaned.csv"
TEST_SET_PATH_FULL = TEST_SET_PATH + "full.csv"
TEST_SET_ALL_PREPROCESSED = TEST_SET_PATH + "all_preprocessed"
TEST_SET_NUMERICAL = TEST_SET_PATH + "numerical"
TEST_SET_LEMMATIZED = TEST_SET_PATH + "lemmatized"
TEST_SET_RAW = TEST_SET_PATH + "raw"

""" MODELS """
MODEL_PATH = "models/"

""" Xgboost """
XGB_PATH = MODEL_PATH + "xgb_model.json"
XGB_DEFAULT_PATH = MODEL_PATH + "default_xgb_model.json"

""" Weak labels """
WEAK_LABELS_PATH = DATA_PATH + "weak_labels.csv"

""" Bert """
BERT_PATH = DATA_PATH + "bert/"
BERT_SUPERVISED_PATH = BERT_PATH + "supervised/"
BERT_WEAK_SUPERVISED_PATH = BERT_PATH + "weak_supervised/"

BERT_SUPERVISED_TRAIN_PATH = BERT_SUPERVISED_PATH + "sup_train_bert"
BERT_SUPERVISED_VAL_PATH = BERT_SUPERVISED_PATH + "sup_val_bert"
BERT_SUPERVISED_TEST_PATH = BERT_SUPERVISED_PATH + "sup_test_bert"

BERT_WEAK_SUPERVISED_TRAIN_PATH = BERT_WEAK_SUPERVISED_PATH + "weak_train_bert"
BERT_WEAK_SUPERVISED_WEAK_TRAIN_PATH = BERT_WEAK_SUPERVISED_PATH + "weak_train_only_weak_bert"
BERT_WEAK_SUPERVISED_VAL_PATH = BERT_WEAK_SUPERVISED_PATH + "weak_val_bert"
BERT_WEAK_SUPERVISED_TEST_PATH = BERT_WEAK_SUPERVISED_PATH + "weak_test_bert"

""" Results"""
RESULTS_PATH = "results/"

""" Labeled and unlabeled """
LABELED_AND_UNLABELED_PATH = DATA_PATH + "labeled_and_unlabeled/"
UNLABELED_TRAIN_RAW = LABELED_AND_UNLABELED_PATH + "unlabeled_train_raw"
UNLABELED_TRAIN_LEMMA = LABELED_AND_UNLABELED_PATH + "unlabeled_train_lemma"
LABELED_VALIDATION_RAW = LABELED_AND_UNLABELED_PATH + "labeled_validation_raw"
LABELED_VALIDATION_LEMMA = LABELED_AND_UNLABELED_PATH + "labeled_validation_lemma"
MIXED_TRAIN_LEMMA = LABELED_AND_UNLABELED_PATH + "mixed_train_lemma"
MIXED_TRAIN_RAW = LABELED_AND_UNLABELED_PATH + "mixed_train_raw"

""" TF-IDF """
TF_IDF_PATH = DATA_PATH + "tf-idf/"
TF_IDF_SUPERVISED_PATH = TF_IDF_PATH + "supervised/"
TF_IDF_WEAK_SUPERVISED_PATH = TF_IDF_PATH + "weak_supervised/"

TF_IDF_SUPERVISED_TRAIN_PATH = TF_IDF_SUPERVISED_PATH + "sup_train_tf_idf"
TF_IDF_SUPERVISED_VAL_PATH = TF_IDF_SUPERVISED_PATH + "sup_val_tf_idf"
TF_IDF_SUPERVISED_TEST_PATH = TF_IDF_SUPERVISED_PATH + "sup_test_tf_idf"

TF_IDF_WEAK_SUPERVISED_TRAIN_PATH = TF_IDF_WEAK_SUPERVISED_PATH + "weak_train_tf_idf"
TF_IDF_WEAK_SUPERVISED_VAL_PATH = TF_IDF_WEAK_SUPERVISED_PATH + "weak_val_tf_idf"
TF_IDF_WEAK_SUPERVISED_TEST_PATH = TF_IDF_WEAK_SUPERVISED_PATH + "weak_test_tf_idf"
