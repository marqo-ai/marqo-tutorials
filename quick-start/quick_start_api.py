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

# Delete the index if it already exists
try:
    mq.index("movies-index").delete()
except:
    pass

#####################################################
### STEP 3. Create a Marqo Index
#####################################################

# Create an index - Using this model: https://huggingface.co/intfloat/e5-base-v2
mq.create_index("movies-index", model="hf/e5-base-v2")

#####################################################
### STEP 4. Adding Documents to the Marqo Index
#####################################################

# Add documents (movie descriptions) to the index
mq.index("movies-index").add_documents(
    [
        {
            "Title": "Inception",   # Title of the movie
            "Description": "A mind-bending thriller about dream invasion and manipulation.",   # Movie description
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
    # Specifies which fields of the documents should be used to generate vectors. In this case, 'Description'.
    tensor_fields=["Description"],
)

#####################################################
### STEP 5. Search!
#####################################################

# Perform a search query on the index
results = mq.index("movies-index").search(
        # Our query
    q="Which movie is about space exploration?"
)

# Print the search results
for result in results['hits']:
    print(f"Title: {result['Title']}, Description: {result['Description']}. Score: {result['_score']}")
    
#####################################################
### STEP 6. Clean Up (If Needed)
#####################################################

# (Optional) delete index if needed 
mq.index("movies-index").delete()