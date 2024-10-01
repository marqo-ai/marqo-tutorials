from marqtune.client import Client
from marqtune.enums import DatasetType, InstanceType
from urllib.request import urlopen
import gzip
import json
import uuid
import os

# Suffix is used just to make the dataset and model names unique
suffix = str(uuid.uuid4())[:8]
print(f"Using suffix={suffix} for this walkthrough")

# Set up Marqtune Client

# To find your API Key, go to Marqo Cloud and click 'API Keys' from the lefthand side navigation bar or visit https://www.marqo.ai/blog/finding-my-marqo-api-key
marqtune_client = Client(url="https://marqtune.marqo.ai", api_key=api_key)

# Downloading the data files needed for this walkthrough
print("Downloading data files:")
# The path to our datasets
base_path = (
    "https://marqo-gcl-public.s3.us-west-2.amazonaws.com/marqtune_test/datasets/v1"
)
# Our training and evaluation data
training_data = "gs_100k_training.csv"
eval_data = "gs_25k_eval.csv"

# Download the datasets
open(training_data, "w").write(
    gzip.open(urlopen(f"{base_path}/{training_data}.gz"), "rb").read().decode("utf-8")
)
open(eval_data, "w").write(
    gzip.open(urlopen(f"{base_path}/{eval_data}.gz"), "rb").read().decode("utf-8")
)

# To be able to create datasets in Marqtune, we first need to identify the columns in the CSVs 
# as well as their types by defining a data schema
data_schema = {
    "query": "text",
    "title": "text",
    "image": "image_pointer",
    "score": "score",
}

# Create the training dataset
training_dataset_name = f"{training_data}-{suffix}"
print(f"Creating training dataset ({training_dataset_name}):")
training_dataset = marqtune_client.create_dataset(
    dataset_name=training_dataset_name,
    file_path=training_data,
    dataset_type=DatasetType.TRAINING,
    data_schema=data_schema,
    query_columns=["query"],
    result_columns=["title", "image"],
    # setting wait_for_completion=True will make this a blocking call and will also print logs interactively
    wait_for_completion=True,
)

# Similarly we create the Evaluation dataset
eval_dataset_name = f"{eval_data}-{suffix}"
print(f"Creating evaluation dataset ({eval_dataset_name}):")
eval_dataset = marqtune_client.create_dataset(
    dataset_name=eval_dataset_name,
    file_path=eval_data,
    dataset_type=DatasetType.EVALUATION,
    data_schema=data_schema,
    query_columns=["query"],
    result_columns=["title", "image"],
    wait_for_completion=True,
)

# Setup training hyperparameters
training_params = {
    "leftKeys": ["query"],
    "leftWeights": [1],
    "rightKeys": ["image", "title"],
    "rightWeights": [0.9, 0.1],
    "weightKey": "score",
    "epochs": 5,
}

# Define base model features
base_model = "ViT-B-32"
base_checkpoint = "laion2b_s34b_b79k"
model_name = f"{training_data}-model-{suffix}"
print(f"Training a new model ({model_name}):")
tuned_model = marqtune_client.train_model(
    dataset_id=training_dataset.dataset_id,
    model_name=f"{training_data}-model-{suffix}",
    instance_type=InstanceType.BASIC,
    base_model=f"Marqo/{base_model}.{base_checkpoint}",
    hyperparameters=training_params,
    wait_for_completion=True,
)

# Define evaluation parameters
eval_params = {
    "leftKeys": ["query"],
    "leftWeights": [1],
    "rightKeys": ["image", "title"],
    "rightWeights": [0.9, 0.1],
    "weightKey": "score",
}

# Evaluating the base model using Marqtune's evaluate feature
print("Evaluating the base model:")
base_model_eval = marqtune_client.evaluate(
    dataset_id=eval_dataset.dataset_id,
    model=f"Marqo/{base_model}.{base_checkpoint}",
    hyperparameters=eval_params,
    wait_for_completion=True,
)

print("Evaluating the tuned model:")
tuned_model_id = tuned_model.model_id
tuned_checkpoint = tuned_model.describe()["checkpoints"][-1]
tuned_model_eval = marqtune_client.evaluate(
    dataset_id=eval_dataset.dataset_id,
    model=f"{tuned_model_id}/{tuned_checkpoint}",
    hyperparameters=eval_params,
    wait_for_completion=True,
)

# Convenience function to inspect evaluation logs and extract the results
def print_eval_results(description, evaluation):
    results = next(
        (
            json.loads(log["message"][index:].replace("'", '"'))
            for log in evaluation.logs()[-10:]
            if (index := log["message"].find("{'mAP@1000': ")) != -1
        ),
        None,
    )
    print(description)
    print(json.dumps(results, indent=4))

# Finally, print out the results
print_eval_results("Evaluation results from base model:", base_model_eval)
print_eval_results("Evaluation results from tuned model:", tuned_model_eval)
