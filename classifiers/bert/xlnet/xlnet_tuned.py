import sys

from simpletransformers.config.model_args import ClassificationArgs

from classifiers.bert.supervised_simple_transformers import run_supervised_simple_transformer_on_all_data
from classifiers.bert.weak_simple_transformers import run_weak_simple_transformer_on_all_data

is_weak_supervised = int(sys.argv[1])    # 1 if running weak supervised, 0 if running supervised

if __name__ == '__main__':

    model_name = 'xlnet'
    model_type = 'xlnet-base-cased'

    model_args = ClassificationArgs()

    # Static params
    model_args.eval_batch_size = 16
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_silent = False
    model_args.evaluate_during_training_steps = 1000
    model_args.manual_seed = 4
    model_args.max_seq_length = 256
    model_args.multiprocessing_chunksize = 5000
    model_args.no_cache = True
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.train_batch_size = 16
    model_args.gradient_accumulation_steps = 2
    model_args.train_custom_parameters_only = False

    if is_weak_supervised:
        print("Running tuned weak supervision with ", model_name)

        # Tuned params
        model_args.num_train_epochs = 9
        model_args.learning_rate = 0.00001685

        model_args.output_dir = 'outputs/xlnet_tuned/weak_supervised/'
        model_args.best_model_dir = 'outputs/xlnet_tuned/weak_supervised/best_model'
        run_weak_simple_transformer_on_all_data(model_name, model_type, model_args, is_tuned=True)

    else:
        print("Running tuned supervised with ", model_name)

        # Tuned params
        model_args.num_train_epochs = 9
        model_args.learning_rate = 0.00007139

        model_args.output_dir = 'outputs/xlnet_tuned/supervised/'
        model_args.best_model_dir = 'outputs/xlnet_tuned/supervised/best_model'
        run_supervised_simple_transformer_on_all_data(model_name, model_type, model_args, is_tuned=True)
