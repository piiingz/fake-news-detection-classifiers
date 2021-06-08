import string

import pandas as pd

from handle_datasets.paths import *

pd.set_option('display.expand_frame_repr', False, 'display.max_rows', None)


""" Utils"""


# Generic save CSV function
def save_dataframe_to_csv(df: pd.DataFrame, path: string):
    df.to_csv(path, index=False)


# Generic save Pickle function
def save_dataframe_to_pkl(df, path: string):
    df.to_pickle(path)


""" Save datasets"""


# save dataset 5 - Full dataset with embeddings (To PKL)
def save_dataset_5(df: pd.DataFrame):
    save_dataframe_to_pkl(df, DATASET_PATHS[5] + '.pkl')


# save dataset 6 - Full dataset with numerical, true and weak labels (To PKL)
def save_dataset_6_pkl(df: pd.DataFrame):
    save_dataframe_to_pkl(df, DATASET_PATHS[6] + '.pkl')


# save dataset 6 - Full dataset with numerical, true and weak labels (To CSV)
def save_dataset_6_csv(df: pd.DataFrame):
    save_dataframe_to_csv(df, DATASET_PATHS[6] + '.csv')


# save both pkl and csv
def save_dataset_6(df: pd.DataFrame):
    save_dataset_6_pkl(df)
    save_dataset_6_csv(df)


# Save test and train
def save_numerical_test_train(train_balanced: pd.DataFrame, test_balanced: pd.DataFrame, train_unbalanced: pd.DataFrame):

    save_dataframe_to_pkl(train_balanced, DATASET_PATHS[6] + TRAIN_BALANCED + '.pkl')
    save_dataframe_to_csv(train_balanced, DATASET_PATHS[6] + TRAIN_BALANCED + '.csv')

    save_dataframe_to_pkl(test_balanced, DATASET_PATHS[6] + TEST_BALANCED + '.pkl')
    save_dataframe_to_csv(test_balanced, DATASET_PATHS[6] + TEST_BALANCED + '.csv')

    save_dataframe_to_pkl(train_unbalanced, DATASET_PATHS[6] + TRAIN_UNBALANCED + '.pkl')
    save_dataframe_to_csv(train_unbalanced, DATASET_PATHS[6] + TRAIN_UNBALANCED + '.csv')


# Save BERT preprocessed train, val test for weak
def save_weak_supervised_bert_preprocessed_datasets(train, val, test, weak_train):

    save_dataframe_to_pkl(train, BERT_WEAK_SUPERVISED_TRAIN_PATH + '.pkl')
    save_dataframe_to_csv(train, BERT_WEAK_SUPERVISED_TRAIN_PATH + '.csv')

    save_dataframe_to_pkl(val, BERT_WEAK_SUPERVISED_VAL_PATH + '.pkl')
    save_dataframe_to_csv(val, BERT_WEAK_SUPERVISED_VAL_PATH + '.csv')

    save_dataframe_to_pkl(test, BERT_WEAK_SUPERVISED_TEST_PATH + '.pkl')
    save_dataframe_to_csv(test, BERT_WEAK_SUPERVISED_TEST_PATH + '.csv')

    save_dataframe_to_pkl(weak_train, BERT_WEAK_SUPERVISED_WEAK_TRAIN_PATH + '.pkl')
    save_dataframe_to_csv(weak_train, BERT_WEAK_SUPERVISED_WEAK_TRAIN_PATH + '.csv')


# Save BERT preprocessed train, val test for weak
def save_supervised_bert_preprocessed_datasets(train, val, test):

    save_dataframe_to_pkl(train, BERT_SUPERVISED_TRAIN_PATH + '.pkl')
    save_dataframe_to_csv(train, BERT_SUPERVISED_TRAIN_PATH + '.csv')

    save_dataframe_to_pkl(val, BERT_SUPERVISED_VAL_PATH + '.pkl')
    save_dataframe_to_csv(val, BERT_SUPERVISED_VAL_PATH + '.csv')

    save_dataframe_to_pkl(test, BERT_SUPERVISED_TEST_PATH + '.pkl')
    save_dataframe_to_csv(test, BERT_SUPERVISED_TEST_PATH + '.csv')


# Save labeled and unlabeled datasets after splitting
def save_labeled_and_unlabeled_datasets(unlabeled_train_raw, unlabeled_train_lemma, labeled_validation_raw, labeled_validation_lemma,
                                        mixed_raw, mixed_lemma):

    save_dataframe_to_pkl(unlabeled_train_raw, UNLABELED_TRAIN_RAW + '.pkl')
    save_dataframe_to_csv(unlabeled_train_raw, UNLABELED_TRAIN_RAW + '.csv')

    save_dataframe_to_pkl(unlabeled_train_lemma, UNLABELED_TRAIN_LEMMA + '.pkl')
    save_dataframe_to_csv(unlabeled_train_lemma, UNLABELED_TRAIN_LEMMA + '.csv')

    save_dataframe_to_pkl(labeled_validation_raw, LABELED_VALIDATION_RAW + '.pkl')
    save_dataframe_to_csv(labeled_validation_raw, LABELED_VALIDATION_RAW + '.csv')

    save_dataframe_to_pkl(labeled_validation_lemma, LABELED_VALIDATION_LEMMA + '.pkl')
    save_dataframe_to_csv(labeled_validation_lemma, LABELED_VALIDATION_LEMMA + '.csv')

    save_dataframe_to_pkl(mixed_raw, MIXED_TRAIN_RAW + '.pkl')
    save_dataframe_to_csv(mixed_raw, MIXED_TRAIN_RAW + '.csv')

    save_dataframe_to_pkl(mixed_lemma, MIXED_TRAIN_LEMMA + '.pkl')
    save_dataframe_to_csv(mixed_lemma, MIXED_TRAIN_LEMMA + '.csv')


# Save train, val and test as tf idf vectors including "id" and "label" columns
def save_weak_supervised_tf_idf_datasets(train, val, test):

    save_dataframe_to_pkl(train, TF_IDF_WEAK_SUPERVISED_TRAIN_PATH + '.pkl')
    save_dataframe_to_csv(train, TF_IDF_WEAK_SUPERVISED_TRAIN_PATH + '.csv')

    save_dataframe_to_pkl(val, TF_IDF_WEAK_SUPERVISED_VAL_PATH + '.pkl')
    save_dataframe_to_csv(val, TF_IDF_WEAK_SUPERVISED_VAL_PATH + '.csv')

    save_dataframe_to_pkl(test, TF_IDF_WEAK_SUPERVISED_TEST_PATH + '.pkl')
    save_dataframe_to_csv(test, TF_IDF_WEAK_SUPERVISED_TEST_PATH + '.csv')


def save_supervised_tf_idf_datasets(train, val, test):

    save_dataframe_to_pkl(train, TF_IDF_SUPERVISED_TRAIN_PATH + '.pkl')
    save_dataframe_to_csv(train, TF_IDF_SUPERVISED_TRAIN_PATH + '.csv')

    save_dataframe_to_pkl(val, TF_IDF_SUPERVISED_VAL_PATH + '.pkl')
    save_dataframe_to_csv(val, TF_IDF_SUPERVISED_VAL_PATH + '.csv')

    save_dataframe_to_pkl(test, TF_IDF_SUPERVISED_TEST_PATH + '.pkl')
    save_dataframe_to_csv(test, TF_IDF_SUPERVISED_TEST_PATH + '.csv')

