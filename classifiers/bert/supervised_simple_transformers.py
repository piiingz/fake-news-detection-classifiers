from handle_datasets.load_datasets import load_supervised_bert_tuning, load_all_supervised_bert_train_val_test
from classifiers.bert.simple_transformer import run_simple_transformer_model


def run_supervised_simple_transformer_on_tuning_set(model_name, model_type, model_args, is_tuned):
    train_df, val_df, test_df = load_supervised_bert_tuning()

    run_simple_transformer_model(model_name, model_type, model_args, train_df, val_df, test_df, is_weak=False, is_tuned=is_tuned)


def run_supervised_simple_transformer_on_all_data(model_name, model_type, model_args, is_tuned):
    train_df, val_df, test_df = load_all_supervised_bert_train_val_test()

    run_simple_transformer_model(model_name, model_type, model_args, train_df, val_df, test_df, is_weak=False, is_tuned=is_tuned)
