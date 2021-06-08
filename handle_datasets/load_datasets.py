import string

import pandas as pd

from config import LABELED_LEMMATIZED_COLS, LABELED_RAW_TEXT_COLS, TUNING_SIZE, LABELED_TRAIN_SIZE

from handle_datasets.paths import *

pd.set_option('display.expand_frame_repr', False, 'display.max_rows', None)


"""
Utils
"""


# Generic load CSV function
def load_dataframe_from_csv(path: string, include_columns: list = None, index_column=None):
    df = pd.read_csv(filepath_or_buffer=path, usecols=include_columns, index_col=index_column)

    return df


# Generic load Pickle function
def load_dataframe_from_pkl(path: string, include_columns: list = None):
    df = pd.read_pickle(path)

    if include_columns:
        df = df[include_columns]

    return df


"""
Load functions for preprocessed datasets from preprocessing/weak labeling repo
"""


# Load dataset 6 - Full dataset with numerical, true and weak labels (from PKL)
def load_dataset_6_pkl():
    df = load_dataframe_from_pkl(DATASET_PATHS[6] + '.pkl')

    return df


# Load dataset 6 - Full dataset with numerical, true and weak labels (from CSV)
def load_dataset_6_csv():
    df = load_dataframe_from_csv(DATASET_PATHS[6] + '.csv')

    return df


# Load numerical train set balanced
def load_dataset_6_train_balanced():
    df = load_dataframe_from_pkl(DATASET_PATHS[6] + TRAIN_BALANCED + '.pkl')

    return df


# Load numerical test set balanced
def load_dataset_6_test_balanced():
    df = load_dataframe_from_pkl(DATASET_PATHS[6] + TEST_BALANCED + '.pkl')

    return df


# Load numerical train unbalanced
def load_dataset_6_train_unbalanced():
    df = load_dataframe_from_pkl(DATASET_PATHS[6] + TRAIN_UNBALANCED + '.pkl')

    return df


# Load numerical + lemmatized train set balanced
def load_dataset_6_train_balanced_lemma():
    df = load_dataframe_from_pkl(DATASET_PATHS[6] + TRAIN_BALANCED_LEMMA + '.pkl')

    return df


# Load numerical + lemmatized test (labeled) set balanced
def load_dataset_6_test_balanced_lemma(include: list = None):
    df = load_dataframe_from_pkl(DATASET_PATHS[6] + TEST_BALANCED_LEMMA + '.pkl', include_columns=include)

    return df


# Load numerical + raw train set balanced
def load_dataset_6_train_balanced_raw():
    df = load_dataframe_from_pkl(DATASET_PATHS[6] + TRAIN_BALANCED_RAW + '.pkl')

    return df


# Load numerical + raw test (labeled) set balanced
def load_dataset_6_test_balanced_raw(include: list = None):
    df = load_dataframe_from_pkl(DATASET_PATHS[6] + TEST_BALANCED_RAW + '.pkl', include_columns=include)

    return df


# Load whole preprocessed testset
def load_testset_total():
    df = load_dataframe_from_pkl(TEST_SET_ALL_PREPROCESSED + '.pkl')

    return df


# Load manually created dataset
def load_dataset_test_full_pkl(include: list = None):
    return load_dataframe_from_pkl(TEST_SET_ALL_PREPROCESSED + ".pkl", include_columns=include)


# Load manually created dataset
def load_dataset_test_full_csv(include: list = None):
    return load_dataframe_from_csv(TEST_SET_ALL_PREPROCESSED + ".csv", include_columns=include)


# Load fake and real 
def load_dataset_test_real():
    return load_dataframe_from_csv(TEST_SET_PATH_REAL, index_column=0)


def load_dataset_test_fake():
    return load_dataframe_from_csv(TEST_SET_PATH_FAKE, index_column=0)


def load_weak_labels():
    df = load_dataframe_from_csv(WEAK_LABELS_PATH, index_column=0)
    df['ground_label'] = df['ground_label'].apply(lambda x: 0 if x == -1 else 1)
    df['weak_label'] = df['weak_label'].apply(lambda x: abs(x)-1 if x in [-1, 0] else 1)
    return df


""" LABELED AND UNLABELED """


# Load labeled validation set
def load_labeled_validation_raw():
    return load_dataframe_from_pkl(LABELED_VALIDATION_RAW + '.pkl')


def load_labeled_validation_lemma():
    return load_dataframe_from_pkl(LABELED_VALIDATION_LEMMA + '.pkl')


# Load training set containing mixed labeled train and weak labeled train
def load_mixed_train_raw():
    return load_dataframe_from_pkl(MIXED_TRAIN_RAW + '.pkl')


def load_mixed_train_lemma():
    return load_dataframe_from_pkl(MIXED_TRAIN_LEMMA + '.pkl')


# Load only weak labeled train
def load_weak_labeled_train_raw():
    return load_dataframe_from_pkl(UNLABELED_TRAIN_RAW + '.pkl')


def load_weak_labeled_train_lemma():
    return load_dataframe_from_pkl(UNLABELED_TRAIN_LEMMA + '.pkl')


# Load all lemmatized datasets for weak supervision
def load_weak_supervised_lemmatized_train_val_test():
    train = load_mixed_train_lemma()
    val = load_labeled_validation_lemma()
    test = load_dataset_test_full_pkl(include=LABELED_LEMMATIZED_COLS)

    return train, val, test


# Load all lemmatized datasets for supervised
def load_supervised_lemmatized_train_val_test():
    train = load_dataset_6_test_balanced_lemma(include=LABELED_LEMMATIZED_COLS)[:LABELED_TRAIN_SIZE]
    val = load_labeled_validation_lemma()
    test = load_dataset_test_full_pkl(include=LABELED_LEMMATIZED_COLS)

    return train, val, test


# Load raw dataset to use for bert preprocessing with weak
def load_weak_raw_train_val_test():
    train = load_mixed_train_raw()
    val = load_labeled_validation_raw()
    test = load_dataset_test_full_pkl(include=LABELED_RAW_TEXT_COLS)

    return train, val, test


# Load raw dataset to use for bert preprocessing with supervised
def load_supervised_raw_train_val_test():
    train = load_dataset_6_test_balanced_raw(include=LABELED_RAW_TEXT_COLS)[:LABELED_TRAIN_SIZE]
    val = load_labeled_validation_raw()
    test = load_dataset_test_full_pkl(include=LABELED_RAW_TEXT_COLS)

    return train, val, test


""" TF IDF """


# Load weak supervised tf idf datasets after they have already been generated
def load_weak_supervised_tf_idf_train():
    return load_dataframe_from_pkl(TF_IDF_WEAK_SUPERVISED_TRAIN_PATH + '.pkl')


def load_weak_supervised_tf_idf_val():
    return load_dataframe_from_pkl(TF_IDF_WEAK_SUPERVISED_VAL_PATH + '.pkl')


def load_weak_supervised_tf_idf_test():
    return load_dataframe_from_pkl(TF_IDF_WEAK_SUPERVISED_TEST_PATH + '.pkl')


# Load all weak supervised tf-idf datasets and drop id column
def load_all_weak_supervised_tf_idf_train_val_test():
    train = load_weak_supervised_tf_idf_train().drop(['id'], axis=1)
    val = load_weak_supervised_tf_idf_val().drop(['id'], axis=1)
    test = load_weak_supervised_tf_idf_test().drop(['id'], axis=1)

    return train, val, test


# Load tf idf for tuning weak supervised
def load_weak_tf_idf_tuning():
    train, val, test = load_all_weak_supervised_tf_idf_train_val_test()
    num_train = round(train.shape[0] * TUNING_SIZE)
    return train.head(n=num_train), val, test


# Load supervised tf idf datasets after they have already been generated
def load_supervised_tf_idf_train():
    return load_dataframe_from_pkl(TF_IDF_SUPERVISED_TRAIN_PATH + '.pkl')


def load_supervised_tf_idf_val():
    return load_dataframe_from_pkl(TF_IDF_SUPERVISED_VAL_PATH + '.pkl')


def load_supervised_tf_idf_test():
    return load_dataframe_from_pkl(TF_IDF_SUPERVISED_TEST_PATH + '.pkl')


# Load all supervised tf-idf datasets and drop id column
def load_all_supervised_tf_idf_train_val_test():
    train = load_supervised_tf_idf_train().drop(['id'], axis=1)
    val = load_supervised_tf_idf_val().drop(['id'], axis=1)
    test = load_supervised_tf_idf_test().drop(['id'], axis=1)

    return train, val, test
  
  
# Load tf idf for tuning supervised
def load_supervised_tf_idf_tuning():
    train, val, test = load_all_supervised_tf_idf_train_val_test()
    num_train = round(train.shape[0] * TUNING_SIZE)
    return train.head(n=num_train), val, test


""" BERT """


# Load weak supervised bert datasets after they have already been generated
def load_weak_supervised_bert_train():
    return load_dataframe_from_pkl(BERT_WEAK_SUPERVISED_TRAIN_PATH + '.pkl')


# Load only weak labeled train
def load_weak_supervised_bert_weak_train():
    return load_dataframe_from_pkl(BERT_WEAK_SUPERVISED_WEAK_TRAIN_PATH + '.pkl')


def load_weak_supervised_bert_val():
    return load_dataframe_from_pkl(BERT_WEAK_SUPERVISED_VAL_PATH + '.pkl')


def load_weak_supervised_bert_test():
    return load_dataframe_from_pkl(BERT_WEAK_SUPERVISED_TEST_PATH + '.pkl')


# Load all bert datasets for weak supervised
def load_all_weak_supervised_bert_train_val_test():
    train = load_weak_supervised_bert_train()
    val = load_weak_supervised_bert_val()
    test = load_weak_supervised_bert_test()

    return train, val, test


# Load bert weak supervised datasets for tuning
def load_weak_bert_tuning():
    train = load_weak_supervised_bert_train()
    val = load_weak_supervised_bert_val()
    test = load_weak_supervised_bert_test()

    num_train = round(train.shape[0] * TUNING_SIZE)

    return train.head(n=num_train), val, test


# Load supervised bert datasets after they have already been generated
def load_supervised_bert_train():
    return load_dataframe_from_pkl(BERT_SUPERVISED_TRAIN_PATH + '.pkl')


def load_supervised_bert_val():
    return load_dataframe_from_pkl(BERT_SUPERVISED_VAL_PATH + '.pkl')


def load_supervised_bert_test():
    return load_dataframe_from_pkl(BERT_SUPERVISED_TEST_PATH + '.pkl')


# Load all bert datasets for supervised
def load_all_supervised_bert_train_val_test():
    train = load_supervised_bert_train()
    val = load_supervised_bert_val()
    test = load_supervised_bert_test()

    return train, val, test


# Load bert supervised datasets for tuning
def load_supervised_bert_tuning():
    train = load_supervised_bert_train()
    val = load_supervised_bert_val()
    test = load_supervised_bert_test()

    num_train = round(train.shape[0] * TUNING_SIZE)

    return train.head(n=num_train), val, test

