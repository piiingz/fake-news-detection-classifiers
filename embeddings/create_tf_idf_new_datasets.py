""""
 1. Load train, val and test lemmatized
 2. Fit vectors on train set (one for content and one for title)
 3. Transform val and test with same vectorizer
 4. Save to file
 5. Create load functions for these tf-idf datasets, including one with only tuning portion of train
"""
import pandas as pd

from embeddings.tf_idf_model import get_tf_idf_vectors_train_val_test, concat_title_and_content_vec
from handle_datasets.load_datasets import load_weak_supervised_lemmatized_train_val_test, \
    load_supervised_lemmatized_train_val_test
from handle_datasets.save_datasets import save_weak_supervised_tf_idf_datasets, save_supervised_tf_idf_datasets


def create_tf_idf_for_new_datasets(train_lemmatized, val_lemmatized, test_lemmatized):

    print("Train with lemmatized text shape: ", train_lemmatized.shape)
    print("Val with lemmatized text shape: ", val_lemmatized.shape)
    print("Test with lemmatized text shape: ", test_lemmatized.shape)

    # Create the vectors
    # For title
    train_title = train_lemmatized["title_lemmatized_lowercase_no_stopwords"]
    val_title = val_lemmatized["title_lemmatized_lowercase_no_stopwords"]
    test_title = test_lemmatized["title_lemmatized_lowercase_no_stopwords"]

    train_title_vec, val_title_vec, test_title_vec = get_tf_idf_vectors_train_val_test(train_title, val_title,
                                                                                       test_title, 'title')

    # For content, but first make into lowercase
    train_content = train_lemmatized['content_lemmatized_lowercase_no_stopwords'].apply(lambda x: [word.lower() for word in x])
    val_content = val_lemmatized['content_lemmatized_lowercase_no_stopwords'].apply(lambda x: [word.lower() for word in x])
    test_content = test_lemmatized['content_lemmatized_lowercase_no_stopwords'].apply(lambda x: [word.lower() for word in x])

    train_content_vec, val_content_vec, test_content_vec = get_tf_idf_vectors_train_val_test(train_content, val_content,
                                                                                             test_content, 'content')

    # Concat title and content vectors
    train_vec = concat_title_and_content_vec(train_title_vec, train_content_vec)
    val_vec = concat_title_and_content_vec(val_title_vec, val_content_vec)
    test_vec = concat_title_and_content_vec(test_title_vec, test_content_vec)

    # Add id and label to vector dataframe
    train_total_tf_idf = concat_vectors_with_labels(train_lemmatized, train_vec)
    val_total_tf_idf = concat_vectors_with_labels(val_lemmatized, val_vec)
    test_total_tf_idf = concat_vectors_with_labels(test_lemmatized, test_vec)

    print("Train total shape: ", train_total_tf_idf.shape)
    print("Val total shape: ", val_total_tf_idf.shape)
    print("Test total shape: ", test_total_tf_idf.shape)

    return train_total_tf_idf, val_total_tf_idf, test_total_tf_idf


def concat_vectors_with_labels(original_df, vectors):
    cols = ["id", "label"]
    concat_vec = pd.concat([original_df[cols].reset_index(drop=True), vectors], axis=1)
    return concat_vec


def create_and_save_weak_supervised_tf_idf():
    # Load datasets
    train_lemmatized, val_lemmatized, test_lemmatized = load_weak_supervised_lemmatized_train_val_test()

    # Add tf idf vectors
    train_total_tf_idf, val_total_tf_idf, test_total_tf_idf = create_tf_idf_for_new_datasets(train_lemmatized, val_lemmatized, test_lemmatized)

    # Save to file
    save_weak_supervised_tf_idf_datasets(train_total_tf_idf, val_total_tf_idf, test_total_tf_idf)


def create_and_save_supervised_tf_idf():
    # Load datasets
    train_lemmatized, val_lemmatized, test_lemmatized = load_supervised_lemmatized_train_val_test()

    # Add tf idf vectors
    train_total_tf_idf, val_total_tf_idf, test_total_tf_idf = create_tf_idf_for_new_datasets(train_lemmatized,
                                                                                             val_lemmatized,
                                                                                             test_lemmatized)

    # Save to file
    save_supervised_tf_idf_datasets(train_total_tf_idf, val_total_tf_idf, test_total_tf_idf)
