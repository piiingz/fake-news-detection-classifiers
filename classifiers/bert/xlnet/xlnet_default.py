import sys

from simpletransformers.config.model_args import ClassificationArgs

from classifiers.bert.supervised_simple_transformers import run_supervised_simple_transformer_on_all_data
from classifiers.bert.weak_simple_transformers import run_weak_simple_transformer_on_all_data

is_weak_supervised = int(sys.argv[1])  # 1 if running weak supervised, 0 if running supervised


if __name__ == '__main__':

    model_name = 'xlnet'
    model_type = 'xlnet-base-cased'

    model_args = ClassificationArgs()
    model_args.overwrite_output_dir = True
    model_args.logging_steps = 250
    model_args.evaluate_during_training_verbose = True

    if is_weak_supervised:
        print("Running default weak supervision with ", model_name)

        model_args.output_dir = 'outputs/xlnet_default/weak_supervised/'
        model_args.best_model_dir = 'outputs/xlnet_default/weak_supervised/best_model'
        run_weak_simple_transformer_on_all_data(model_name, model_type, model_args, is_tuned=False)

    else:
        print("Running default supervised with ", model_name)

        model_args.output_dir = 'outputs/xlnet_default/supervised/'
        model_args.best_model_dir = 'outputs/xlnet_default/supervised/best_model'
        run_supervised_simple_transformer_on_all_data(model_name, model_type, model_args, is_tuned=False)

