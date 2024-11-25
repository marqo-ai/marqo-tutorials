from marqo import Client
from pprint import pprint

#####################################################
### STEP 1. Start Marqo Locally
#####################################################

"""
docker pull marqoai/marqo:latest
docker rm -f marqo
docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
"""

####################################################
### STEP 2: Create Marqo Index
####################################################

# Initialize the Marqo Client
mq = Client("http://localhost:8882")

# Name your index
index_name = 'image-search'

# We create the index. Note if it already exists an error will occur
# as you cannot overwrite an existing index. For this reason, we delete
# any existing index 
try:
   mq.index(index_name).delete()
except:
    pass # It's safe to ignore errors if the index does not exist.

# Define settings for the index
settings = {
    "model": "ViT-B/32",   # A Vision Transformer model for image embeddings.
    "treatUrlsAndPointersAsImages": True,   # URLs will be treated as image sources.
}

# Create the index
mq.create_index(index_name, settings_dict=settings)

# ####################################################
# ### STEP 3: Add Images to the Index
# ####################################################

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

# Define a query
query = "A rider on a horse jumping over the barrier"

# Perform a search for this query
search_results = mq.index(index_name).search(query)

# Obtain the top result 
top_result = search_results['hits'][0]
# Print the top result
print(search_results['hits'][0])

####################################################
### STEP 5: Visualize the Output
####################################################

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import urllib.request
from io import BytesIO

# URL of the image
image_url = top_result['image']

# Open the URL and read the image into a Pillow Image object
with urllib.request.urlopen(image_url) as url:
    img = Image.open(BytesIO(url.read()))

# Convert the PIL Image object to a NumPy array for matplotlib
img_array = np.array(img)

# Display the image using matplotlib
plt.imshow(img_array)
plt.axis('off')  # Hide the axis
plt.show()