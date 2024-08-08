#####################################################
### STEP 1. Setting UP
#####################################################

# 1. Sign Up to Marqo Cloud: https://cloud.marqo.ai/
# 2. Get a Marqo API Key: https://www.marqo.ai/blog/finding-my-marqo-api-key

#####################################################
### STEP 2. Create Marqo Client
#####################################################
import marqo

# Create a Marqo client
api_key = "your_api_key"  # replace with your api key (https://www.marqo.ai/blog/finding-my-marqo-api-key)
api_key = "QWOBTukZ58/lUk84V+Vm0e7LS0aAuZ4jc7iCqEhKidi75M8VaS72DphVRAIlB8Al"
mq = marqo.Client("https://api.marqo.ai", api_key=api_key)

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