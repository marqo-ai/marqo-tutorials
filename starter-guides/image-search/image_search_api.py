from marqo import Client
from pprint import pprint

#####################################################
### STEP 1. Start Marqo
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
client = Client("http://localhost:8882")

# Name your index
index_name = 'image-search-api'

# We create the index. Note if it already exists an error will occur
# as you cannot overwrite an existing index. For this reason, we delete
# any existing index 
try:
    client.index(index_name).delete()
except:
    pass

# Define settings for the index
settings = {
    "model": "ViT-B/32",
    "treatUrlsAndPointersAsImages": True,
}

# Create the index
client.create_index(index_name, settings_dict=settings)

# ####################################################
# ### STEP 3: Add Images to the Index
# ####################################################

# We will use 4 images from our examples folder in our GitHub repo: https://github.com/marqo-ai/marqo
documents = [
    {"image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image0.jpg"},
    {"image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg"},
    {"image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg"},
    {"image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg"}
    # Add more documents here 
]

# Add these documents to the index
res = client.index(index_name).add_documents(
    documents,
    client_batch_size=1,
    tensor_fields=["image"]
)

# Print out if you wish to
pprint(res)

####################################################
### STEP 4: Search using Marqo
####################################################

# Define a query
query = "A rider on a horse jumping over the barrier"

# Perform a search for this query
search_results = client.index(index_name).search(query)

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