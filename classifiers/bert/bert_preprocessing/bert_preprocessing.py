import pandas as pd

from config import BERT_N_WORDS, BERT_SAVE_COLS, LABELED_RAW_TEXT_COLS
from handle_datasets.load_datasets import load_weak_raw_train_val_test, load_supervised_raw_train_val_test, \
    load_weak_labeled_train_raw
from handle_datasets.save_datasets import save_weak_supervised_bert_preprocessed_datasets, \
    save_supervised_bert_preprocessed_datasets

pd.options.mode.chained_assignment = None  # default='warn'


def trim_string(x):

    x = x.split(maxsplit=BERT_N_WORDS)
    x = ' '.join(x[:BERT_N_WORDS])

    return x


def create_bert_preprocessed_train_val_test(train_raw, val_raw, test_raw):

    print("Train total shape:", train_raw.shape)
    print("Val total shape:", val_raw.shape)
    print("Test total shape:", test_raw.shape)

    # Extract needed cols
    train_bert = train_raw[LABELED_RAW_TEXT_COLS]
    val_bert = val_raw[LABELED_RAW_TEXT_COLS]
    test_bert = test_raw[LABELED_RAW_TEXT_COLS]

    # Concat title and content into one column with period between
    train_bert['text'] = train_bert['title'] + '. ' + train_bert['content']
    val_bert['text'] = val_bert['title'] + '. ' + val_bert['content']
    test_bert['text'] = test_bert['title'] + '. ' + test_bert['content']

    # Cut length
    train_bert['text'] = train_bert['text'].apply(trim_string)
    val_bert['text'] = val_bert['text'].apply(trim_string)
    test_bert['text'] = test_bert['text'].apply(trim_string)

    # Only use bert columns needed in model (exclude id)
    train = train_bert[BERT_SAVE_COLS]
    val = val_bert[BERT_SAVE_COLS]
    test = test_bert[BERT_SAVE_COLS]

    print("Train columns:", train.columns)
    print("Val columns:", val.columns)
    print("Test columns:", test.columns)

    print("Train bert shape:", train.shape)
    print("Validation bert shape:", val.shape)
    print("Test bert shape:", test.shape)

    print("Train label unique values: ", train["label"].value_counts())
    print("Val label unique values: ", val["label"].value_counts())
    print("Test label unique values: ", test["label"].value_counts())

    return train, val, test


def preprocess_weak_train(weak_train):
    print("Weak train total shape:", weak_train.shape)

    # Extract needed cols
    weak_train_bert = weak_train[LABELED_RAW_TEXT_COLS]

    # Concat title and content into one column with period between
    weak_train_bert['text'] = weak_train_bert['title'] + '. ' + weak_train_bert['content']

    # Cut length
    weak_train_bert['text'] = weak_train_bert['text'].apply(trim_string)

    # Only use bert columns needed in model (exclude id)
    weak_train = weak_train_bert[BERT_SAVE_COLS]

    print("Weak train columns:", weak_train.columns)
    print("Weak train bert shape:", weak_train.shape)
    print("Weak train label unique values: ", weak_train["label"].value_counts())

    return weak_train


def create_and_save_weak_bert():
    # Load raw datasets
    train_raw, val_raw, test_raw = load_weak_raw_train_val_test()
    weak_train = load_weak_labeled_train_raw()

    # Preprocess
    train_bert, val_bert, test_bert = create_bert_preprocessed_train_val_test(train_raw, val_raw, test_raw)
    weak_train_bert = preprocess_weak_train(weak_train)

    # Save
    save_weak_supervised_bert_preprocessed_datasets(train_bert, val_bert, test_bert, weak_train_bert)


def create_and_save_supervised_bert():
    # Load raw datasets
    train_raw, val_raw, test_raw = load_supervised_raw_train_val_test()

    # Preprocess
    train_bert, val_bert, test_bert = create_bert_preprocessed_train_val_test(train_raw, val_raw, test_raw)

    # Save
    save_supervised_bert_preprocessed_datasets(train_bert, val_bert, test_bert)
