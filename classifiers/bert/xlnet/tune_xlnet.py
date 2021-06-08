import logging
import sys

import torch
import wandb
from simpletransformers.classification import ClassificationArgs, ClassificationModel
from sklearn.metrics import accuracy_score

from handle_datasets.load_datasets import load_all_weak_supervised_bert_train_val_test, \
    load_all_supervised_bert_train_val_test

is_weak_supervised = int(sys.argv[1])    # 1 if running weak supervised, 0 if running supervised

cuda_available = torch.cuda.is_available()
print("Cuda: ", cuda_available)

# Loading datasets
if is_weak_supervised:
    train_df, eval_df, test_df = load_all_weak_supervised_bert_train_val_test()
    model_setting = 'weak'
    output_subdir = 'weak_supervised/'

else:
    train_df, eval_df, test_df = load_all_supervised_bert_train_val_test()
    model_setting = 'supervised'
    output_subdir = 'supervised/'

print("Setting: ", model_setting)
print("Train bert shape:", train_df.shape)
print("Validation bert shape:", eval_df.shape)
print("Test bert shape:", test_df.shape)


# Configuring sweep
sweep_config = {
    "name": "xlnet-"+model_setting+"-sweep-batch-16",
    "method": "bayes",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "num_train_epochs": {"min": 1, "max": 10},
        "learning_rate": {"min": 0, "max": 4e-4},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3, },
}

sweep_id = wandb.sweep(sweep_config, project="master-thesis")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# Configure model args
model_args = ClassificationArgs()
model_args.eval_batch_size = 16
model_args.evaluate_during_training = True
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = 1000
model_args.learning_rate = 4e-4
model_args.manual_seed = 4
model_args.max_seq_length = 256
model_args.multiprocessing_chunksize = 5000
model_args.no_cache = True
model_args.no_save = True
model_args.num_train_epochs = 2
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 2
model_args.train_custom_parameters_only = False

model_args.output_dir = 'outputs/tune_xlnet/' + output_subdir
model_args.best_model_dir = 'outputs/tune_xlnet/' + output_subdir + 'best_model'
model_args.wandb_project = "master-thesis"


# Training and tuning model with wandb
def train():
    # Initialize a new wandb run
    wandb.init()

    # Create a TransformerModel
    model = ClassificationModel(
        "xlnet",
        "xlnet-base-cased",
        use_cuda=cuda_available,
        args=model_args,
        sweep_config=wandb.config,
    )

    # Train the model
    model.train_model(
        train_df,
        eval_df=eval_df,
        accuracy=lambda truth, predictions: accuracy_score(
            truth, [round(p) for p in predictions]
        ),
    )

    # Sync wandb
    wandb.join()


wandb.agent(sweep_id, train)
