#####################################################
### STEP 1. Start Marqo
#####################################################

# 1. Marqo requires Docker. To install Docker go to Docker 
# Docs and install for your operating system.

# 2. Once Docker is installed, you can use it to run Marqo. 
# First, open the Docker application and then head to your 
# terminal and enter the following:

"""
docker pull marqoai/marqo:latest
docker rm -f marqo
docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
"""

#####################################################
### STEP 2. Create Marqo Client
#####################################################
import marqo

# Create a Marqo client
mq = marqo.Client(url="http://localhost:8882")

#####################################################
### STEP 3. Create a Marqo Index & Add Documents
#####################################################

# Name your index
index_name = "text-search-open-source"

# Delete the movie index if it already exists (housekeeping)
try:
    mq.index(index_name).delete()
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
