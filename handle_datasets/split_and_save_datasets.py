
import pandas as pd

from config import RANDOM_SEED, LABELED_RAW_TEXT_COLS, LABELED_LEMMATIZED_COLS, LABELED_VALIDATION_ABSOLUTE_SIZE, \
    LABELED_TRAIN_SIZE, WEAK_NUMBER_OF_ARTICLES_PER_CLASS
from handle_datasets.load_datasets import load_dataset_6_train_balanced_raw, load_dataset_6_train_balanced_lemma, \
    load_dataset_6_test_balanced_raw, load_dataset_6_test_balanced_lemma, load_weak_labels
from handle_datasets.save_datasets import save_labeled_and_unlabeled_datasets

"""
1. Load train and test lemmatized and raw, from now on called unlabeled and labeled
2. Extract 5% of total from unlabeled for validation/model selection with true labels
3. Append weak labels to the remaining part of unlabeled
4. Mix labeled train with true labels and unlabeled with weak labels
5. Save the following separately: validation set (5%), unlabeled minus validation set (75%), mixed labeled + unlabeled (95%)
6. Make functions to load these saved datasets (do not select id)
"""

pd.options.mode.chained_assignment = None  # default='warn'


def load_unlabeled_and_labeled_raw_and_lemma():

    unlabeled_raw = load_dataset_6_train_balanced_raw()
    unlabeled_lemma = load_dataset_6_train_balanced_lemma()
    labeled_raw = load_dataset_6_test_balanced_raw()[:LABELED_TRAIN_SIZE]
    labeled_lemma = load_dataset_6_test_balanced_lemma()[:LABELED_TRAIN_SIZE]

    # Discard numerical cols
    unlabeled_raw_without_numerical = unlabeled_raw[LABELED_RAW_TEXT_COLS]
    unlabeled_lemma_without_numerical = unlabeled_lemma[LABELED_LEMMATIZED_COLS]
    labeled_raw_without_numerical = labeled_raw[LABELED_RAW_TEXT_COLS]
    labeled_lemma_without_numerical = labeled_lemma[LABELED_LEMMATIZED_COLS]

    return unlabeled_raw_without_numerical, unlabeled_lemma_without_numerical, labeled_raw_without_numerical, \
        labeled_lemma_without_numerical


def create_labeled_validation_dataset(unlabeled_total):

    print("Unlabeled total shape: ", unlabeled_total.shape)

    # Take out validation set from unlabeled
    total_size = unlabeled_total.shape[0]
    validation_size = LABELED_VALIDATION_ABSOLUTE_SIZE
    train_size = total_size - validation_size
    print(total_size, validation_size, train_size)

    labeled_validation = unlabeled_total.tail(validation_size)
    unlabeled_train = unlabeled_total.head(train_size)

    # Remove true labels from unlabeled portion
    unlabeled_train = unlabeled_train.drop(['label'], axis=1)

    # Print shapes and columns
    print("Unlabeled train shape: ", unlabeled_train.shape)
    print("Validation shape: ", labeled_validation.shape)
    print("Columns:")
    print("Unlabaled train columns: ", unlabeled_train.columns)
    print("Validation columns: ", labeled_validation.columns)

    return unlabeled_train, labeled_validation


def append_weak_labels(unlabeled_train):
    # Load weak labels
    weak_labels = load_weak_labels()

    print("Unlabeled train columns before adding weak: ", unlabeled_train.columns)

    # Append to unlabeled df, and remove abstain
    unlabeled_full = pd.merge(unlabeled_train, weak_labels, on='id')
    unlabeled_non_abstain = unlabeled_full[unlabeled_full['weak_label'] != -1]
    print("Non abstain: ", unlabeled_non_abstain.shape)
    unlabeled_non_abstain["label"] = unlabeled_non_abstain['weak_label'].astype(int)

    # Sort out only the most certain articles of each class
    unlabeled_sorted = unlabeled_non_abstain.sort_values("prob_label")
    unlabeled_filtered = unlabeled_sorted.head(WEAK_NUMBER_OF_ARTICLES_PER_CLASS).append(unlabeled_sorted.tail(
        WEAK_NUMBER_OF_ARTICLES_PER_CLASS))

    # Shuffle
    unlabeled_filtered = unlabeled_filtered.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Drop prob label and other cols not needed in end models
    unlabeled_filtered = unlabeled_filtered.drop(['weak_label', 'ground_label', 'prob_label'], axis=1)
    print("Unlabeled train columns after adding weak: ", unlabeled_filtered.columns)

    return unlabeled_filtered


def mix_labeled_train_and_unlabeled(labeled_train, unlabeled_train):

    print("Labeled train shape: ", labeled_train.shape)
    print("Unlabeled train shape: ", unlabeled_train.shape)

    # Concat
    mixed = pd.concat([labeled_train, unlabeled_train], ignore_index=True)

    # Shuffle
    mixed_shuffled = mixed.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print("Mixed shape: ", mixed_shuffled.shape)

    return mixed_shuffled


def save_labeled_and_mixed_dataset(unlabeled_train_raw, unlabeled_train_lemma, labeled_validation_raw, labeled_validation_lemma,
                                   mixed_raw, mixed_lemma):

    save_labeled_and_unlabeled_datasets(unlabeled_train_raw, unlabeled_train_lemma, labeled_validation_raw,
                                        labeled_validation_lemma, mixed_raw, mixed_lemma)


def split_and_save_labeled_and_unlabeled():

    # Load datasets
    unlabeled_raw, unlabeled_lemma, labeled_train_raw, labeled_train_lemma = load_unlabeled_and_labeled_raw_and_lemma()

    # Split unlabeled into unlabeled train and labeled validation
    print("Raw:")
    unlabeled_train_raw, labeled_validation_raw = create_labeled_validation_dataset(unlabeled_raw)
    print("Lemma:")
    unlabeled_train_lemma, labeled_validation_lemma = create_labeled_validation_dataset(unlabeled_lemma)

    # Check that ids are the same
    print("Ids are the same, unlabeled train: ", unlabeled_train_raw["id"].equals(unlabeled_train_lemma["id"]))
    print("Ids are the same, validation set: ", labeled_validation_raw["id"].equals(labeled_validation_lemma["id"]))

    # Add weak labels to the unlabeled sets
    print("Raw:")
    unlabeled_raw_with_weak_labels = append_weak_labels(unlabeled_train_raw)
    print("Lemma:")
    unlabeled_lemma_with_weak_labels = append_weak_labels(unlabeled_train_lemma)

    # Mix and shuffle labeled and unlabeled
    print("Raw:")
    mixed_raw = mix_labeled_train_and_unlabeled(labeled_train_raw, unlabeled_raw_with_weak_labels)
    print("Lemma:")
    mixed_lemma = mix_labeled_train_and_unlabeled(labeled_train_lemma, unlabeled_lemma_with_weak_labels)

    # Save to file
    save_labeled_and_mixed_dataset(unlabeled_raw_with_weak_labels, unlabeled_lemma_with_weak_labels, labeled_validation_raw,
                                   labeled_validation_lemma, mixed_raw, mixed_lemma)
