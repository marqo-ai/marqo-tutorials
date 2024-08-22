#####################################################
### STEP 1. Setting UP
#####################################################

# 1. Sign Up to Marqo Cloud: https://cloud.marqo.ai/
# 2. Get a Marqo API Key: https://www.marqo.ai/blog/finding-my-marqo-api-key

#####################################################
### STEP 2. Create Marqo Client
#####################################################
from marqo import Client

# Replace this with your API Key
api_key = "your_api_key"

# Set up the Client
mq = Client(
    "https://api.marqo.ai", 
    api_key=api_key
)

#####################################################
### STEP 3. Create a Marqo Index & Add Documents
#####################################################

# Name your index
index_name = 'text-search-cloud'

# We create the index. Note if it already exists an error will occur
# as you cannot overwrite an existing index. For this reason, we delete
# any existing index 
try:
    mq.delete_index(index_name)
except:
    pass

# Create the movie index 
mq.create_index(index_name, model="hf/e5-base-v2")

# Add documents (movie descriptions) to the index
mq.index(index_name).add_documents(
    [
        {
            "Title": "Inception",
            "Description": "A mind-bending thriller about dream invasion and manipulation.",
        },
        {
            "Title": "Shrek",
            "Description": "An ogre's peaceful life is disrupted by a horde of fairy tale characters who need his help.",
        },
        {
            "Title": "Interstellar",
            "Description": "A team of explorers travel through a wormhole in space to ensure humanity's survival.",
        },
        {
            "Title": "The Martian",
            "Description": "An astronaut becomes stranded on Mars and must find a way to survive.",
        },
    ],
    tensor_fields=["Description"],
)

#####################################################
### STEP 4. Search with Marqo
#####################################################

# Perform a search query on the index
results = mq.index(index_name).search(
    q="Which movie is about space exploration?"
)

# Print the search results
for result in results['hits']:
    print(f"Title: {result['Title']}, Description: {result['Description']}. Score: {result['_score']}")

