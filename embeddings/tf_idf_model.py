""""
TF-IDF vectorizer model
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import TITLE_MAX_FEATURES, CONTENT_MAX_FEATURES


def get_tf_idf_vectors_train_val_test(train_words, val_words, test_words, prefix):

    """
    prefix: Either title or content
    """

    vectorizer = get_vectorizer(prefix)

    # Fit vectors to train words
    train_vectors = vectorizer.fit_transform(train_words)

    # Transform validation and test data without fitting
    val_vectors = vectorizer.transform(val_words)
    test_vectors = vectorizer.transform(test_words)

    # Make into dense list
    feature_names = vectorizer.get_feature_names()
    train_dense = train_vectors.todense()
    train_dense_list = train_dense.tolist()

    val_dense = val_vectors.todense()
    val_dense_list = val_dense.tolist()

    test_dense = test_vectors.todense()
    test_dense_list = test_dense.tolist()

    # Make into df with prefix
    train_vec = pd.DataFrame(train_dense_list, columns=feature_names)
    train_vec = train_vec.add_prefix(prefix + '_')

    val_vec = pd.DataFrame(val_dense_list, columns=feature_names)
    val_vec = val_vec.add_prefix(prefix + '_')

    test_vec = pd.DataFrame(test_dense_list, columns=feature_names)
    test_vec = test_vec.add_prefix(prefix + '_')

    return train_vec, val_vec, test_vec


def get_vectorizer(type):
    title_vectorizer = TfidfVectorizer(tokenizer=lambda i: i, lowercase=False, max_features=TITLE_MAX_FEATURES,
                                       ngram_range=(1, 2))
    content_vectorizer = TfidfVectorizer(tokenizer=lambda i: i, lowercase=False, max_features=CONTENT_MAX_FEATURES,
                                         ngram_range=(1, 2))

    if type == "title":
        return title_vectorizer

    elif type == "content":
        return content_vectorizer

    else:
        raise ValueError("Vectorizer type must be either 'title' or 'content'")


def concat_title_and_content_vec(title_vec, content_vec):
    concat_vec = pd.concat([title_vec, content_vec], axis=1)

    return concat_vec
