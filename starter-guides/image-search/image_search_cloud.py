#####################################################
### STEP 1. Obtain Marqo API Key
#####################################################

# 1. Sign Up to Marqo Cloud: https://cloud.marqo.ai/
# 2. Get a Marqo API Key: https://www.marqo.ai/blog/finding-my-marqo-api-key
# Replace this with your Marqo API key:
api_key = "your_api_key"

####################################################
### STEP 2: Initialize Marqo Client
####################################################

from marqo import Client

# Initialize the Marqo client with the API URL and your API key.
# The API key is required to authenticate requests to the Marqo API.
mq = Client(
    "https://api.marqo.ai", 
    api_key=api_key
)

####################################################
### STEP 3: Create a Marqo Index
####################################################

# Define a unique name for your index.
index_name = 'image-search'

# Check if an index with the same name already exists and delete it if necessary.
# This avoids errors when trying to create an index with an existing name.
try:
    mq.index(index_name).delete()
except:
    pass  # It's safe to ignore errors if the index does not exist.

# Configure the index settings, such as the model to use for embedding images,
# whether URLs should be treated as images, and the inference hardware type.
settings = {
    "model": "ViT-B/32",  # A Vision Transformer model for image embeddings.
    "treatUrlsAndPointersAsImages": True,  # URLs will be treated as image sources.
    "inferenceType": "marqo.GPU",  # Use GPU for faster inference.
}

# Create the index with the specified settings.
mq.create_index(index_name, settings_dict=settings)

####################################################
### STEP 4: Add Images to the Index
####################################################

from pprint import pprint

# Add a list of documents containing image URLs and associated descriptions to the index.
# These images are hosted on the Marqo GitHub repository.
documents = [
    {"title": "a woman on her phone taking a photo", "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg"},
    {"title": "a horse and rider jumping over a fence", "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg"},
    {"title": "an aeroplane and the moon", "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"},
    {"title": "man stood by a traffic light", "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg"}
]

# Add the documents to the index with specific mappings and tensor fields.
# The mapping 'image_title_multimodal' combines text and image features with specified weights.
res = mq.index(index_name).add_documents(
    documents,
    client_batch_size=1,  # Add documents one at a time for simplicity.
    mappings={
        "image_title_multimodal": {
            "type": "multimodal_combination",  # Combine text and image modalities.
            "weights": {"title": 0.1, "image": 0.9},  # Assign higher importance to image data.
        }
    },
    tensor_fields=["image_title_multimodal"],  # Specify fields to generate tensors for.
)

# Print the result of adding documents to verify success.
pprint(res)

####################################################
### STEP 4: Search using Marqo
####################################################

# Define a natural language query to search the indexed documents.
query = "A rider on a horse jumping over the barrier"

# Execute the search and retrieve the results.
search_results = mq.index(index_name).search(query)

# Extract the top result from the search results.
top_result = search_results['hits'][0]

# Print the top result for debugging or verification purposes.
print(search_results['hits'][0])

####################################################
### STEP 5: Visualize the Output
####################################################

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import urllib.request
from io import BytesIO

# Get the image URL from the top search result.
image_url = top_result['image']

# Download the image from the URL and load it as a PIL Image object.
with urllib.request.urlopen(image_url) as url:
    img = Image.open(BytesIO(url.read()))

# Convert the PIL Image object to a NumPy array for compatibility with matplotlib.
img_array = np.array(img)

# Display the image using matplotlib without axis labels or ticks.
plt.imshow(img_array)
plt.axis('off')  # Hide the axis for a cleaner display.
plt.show()
