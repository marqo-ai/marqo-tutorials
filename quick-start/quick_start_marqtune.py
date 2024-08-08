#####################################################
### STEP 1. Setting Up
#####################################################

# 1. Sign Up to Marqo Cloud: https://cloud.marqo.ai/
# 2. Get a Marqo API Key: https://www.marqo.ai/blog/finding-my-marqo-api-key

#####################################################
### STEP 2. Set up Marqtune Client
#####################################################

from marqtune import Client
from marqtune.enums import DatasetType, ModelType, InstanceType

# Define Marqo Cloud API Key. For information visit: https://marqo.ai/blog/finding-my-marqo-api-key
api_key = "your_api_key"

marqtune_client = Client(
    "https://marqtune.marqo.ai",
    api_key=api_key
)

# Specify path to your csv. We will use a small dataset which you can
# download from: https://drive.google.com/file/d/12rzhI4DE-x7GVoKC8yRDjmgwA99Zf4zV/view?usp=sharing
input_data_path = "quick-start/marqtune_wclip_pairs.csv" 

# Specify model name
model_name = "quick_start_marqtune"

#####################################################
### STEP 3. Creating Dataset
#####################################################

# Define dataset schema. These headings MUST match those in your csv.
dataset_schema = {
    "text-1": "text", "text-2": "text", "image-1": "image_pointer", "score": "score"
}

# Creating dataset
print(f"Creating dataset with name: {model_name}_dataset")
dataset = marqtune_client.create_dataset(
    model_name + "_dataset", input_data_path, data_schema=dataset_schema, dataset_type=DatasetType.TRAINING
)

#####################################################
### STEP 3. Fine-Tuning Base Model
#####################################################

# Define base model and checkpoints to perform fine-tuning on
base_model = "ViT-B-32"
base_checkpoint = "laion400m_e31"

# Define training task parameters 
train_task_params = {
    "warmup": 0,
    "epochs": 5,
    "lr": 2e-05,
    "precision": "amp",
    "workers": 2,
    "batchSize": 256,
    "wd": 0.02,
    "weightedLoss": "ce",
    "rightKeys": ["text-2"],
    "leftWeights": [1],
    "rightWeights": [1],
    "leftKeys": ["text-1"],
    "weightKey": "score"
}

# Fine-tune the base model 
tuned_model = marqtune_client.train_model(
    dataset.dataset_id,
    model_name,
    base_model,
    base_checkpoint,
    ModelType.OPEN_CLIP,
    instance_type=InstanceType.BASIC,
    hyperparameters=train_task_params
)

# Download the model in '.pt' format 
print("Downloading model and logs")
marqtune_client.model(model_id=tuned_model.model_id).download()