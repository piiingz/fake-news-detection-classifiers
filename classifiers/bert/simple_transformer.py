from datetime import datetime

import torch
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score

from handle_datasets.paths import RESULTS_PATH

cuda_available = torch.cuda.is_available()


def run_simple_transformer_model(model_name, model_type, model_args, train_df, val_df, test_df, is_weak, is_tuned):

    print("Train: ", train_df.columns, train_df.shape)
    print("Val: ", val_df.columns, val_df.shape)
    print("Test: ", test_df.columns, test_df.shape)
    print("Cuda:", cuda_available)

    print("Params: ")
    print("Epochs: ", model_args.num_train_epochs)
    print("Learning rate: ", model_args.learning_rate)

    model = ClassificationModel(model_name, model_type, use_cuda=cuda_available, args=model_args)

    model.train_model(
        train_df,
        eval_df=val_df,
        accuracy=lambda truth, predictions: accuracy_score(
            truth, [round(p) for p in predictions]
        ),
    )

    evaluate_and_write_results_to_file(model, model_name, model_args.output_dir, val_df, test_df, is_weak, is_tuned)


def load_and_evaluate_model(model_name, model_path, val, test, is_weak, is_tuned):

    # Load local model
    model = ClassificationModel(model_name, model_path, use_cuda=cuda_available)
    evaluate_and_write_results_to_file(model, model_name, model_path, val, test, is_weak, is_tuned)


def evaluate_and_write_results_to_file(model, model_name, model_path, val, test, is_weak, is_tuned):
    date = datetime.today().strftime('%Y-%m-%d')

    if is_weak:
        model_type = '-weak'
    else:
        model_type = '-supervised'

    if is_tuned:
        model_type = model_type + '-tuned'
    else:
        model_type = model_type + '-default'

    print("Validation")
    val_result, val_model_outputs, val_wrong_predictions = model.eval_model(val)
    print(val_result)

    path = RESULTS_PATH + model_name + model_type + "-val-" + date + ".txt"
    write_bert_results(path, model_name, model_path, val.shape, val_result, val_model_outputs,
                       val_wrong_predictions)

    print("Test")
    test_result, test_model_outputs, test_wrong_predictions = model.eval_model(test)
    print(test_result)

    path = RESULTS_PATH + model_name + model_type + "-test-" + date + ".txt"
    write_bert_results(path, model_name, model_path, test.shape, test_result, test_model_outputs,
                       test_wrong_predictions)


def write_bert_results(path, model_name, model_path, shape, result, model_output, wrong_predictions):
    with open(path, 'w') as f:
        f.write("\nModel: {}".format(model_name))
        f.write("\nModel path: {}".format(model_path))
        f.write("\nDataset shape: {}".format(shape))
        f.write("\nResult: {}".format(result))
        f.write("\nModel outputs: {}".format(model_output))
        f.write("\nWrong predictions: {}".format(wrong_predictions))
