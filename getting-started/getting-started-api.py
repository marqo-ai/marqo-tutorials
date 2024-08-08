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
### STEP 2. Simple Search with Marqo
#####################################################

import marqo

# Create a Marqo client
mq = marqo.Client(url="http://localhost:8882")

# Housekeeping - Delete the index if it already exists
try:
    mq.index("my-first-index").delete()
except:
    pass

# Create the index
mq.create_index("my-first-index", model="hf/e5-base-v2")

# Add documents to the index
mq.index("my-first-index").add_documents(
    [
        {
            "Title": "The Travels of Marco Polo",
            "Description": "A 13th-century travelogue describing Polo's travels",
        },
        {
            "Title": "Extravehicular Mobility Unit (EMU)",
            "Description": "The EMU is a spacesuit that provides environmental protection, "
            "mobility, life support, and communications for astronauts",
            "_id": "article_591",
        },
    ],
    tensor_fields=["Description"],
)

# Obtain results for a specific query
results = mq.index("my-first-index").search(
    q="What is the best outfit to wear on the moon?"
)

# Print the results
import pprint

pprint.pprint(results)

#####################################################
###          Other Basic Operations
#####################################################

# Get a document
results = mq.index("my-first-index").get_document(document_id="article_591")
print(results)

# Get index stats
results = mq.index("my-first-index").get_stats()
print(results)

# Perform lexical search
results = mq.index("my-first-index").search("marco polo", search_method="LEXICAL")
print(results)

# Perform hybrid search
results = mq.index("my-first-index").search("marco polo", search_method="HYBRID")
print(results)

# # (Optionally) delete the index
# # mq.index("my-first-index").delete()

#####################################################
###         Multimodal and Cross Modal Search
#####################################################

settings = {
    "treat_urls_and_pointers_as_images": True,  # allows us to find an image file and index it
    "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
}

try:
    mq.index("my-multimodal-index").delete()
except:
    pass

response = mq.create_index("my-multimodal-index", **settings)

response = mq.index("my-multimodal-index").add_documents(
    [
        {
            "My_Image": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Hipop%C3%B3tamo_%28Hippopotamus_amphibius%29%2C_parque_nacional_de_Chobe%2C_Botsuana%2C_2018-07-28%2C_DD_82.jpg/640px-Hipop%C3%B3tamo_%28Hippopotamus_amphibius%29%2C_parque_nacional_de_Chobe%2C_Botsuana%2C_2018-07-28%2C_DD_82.jpg",
            "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
            "_id": "hippo-facts",
        }
    ],
    tensor_fields=["My_Image"],
    image_download_headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
)

results = mq.index("my-multimodal-index").search("animal")

print(results)

#####################################################
###         Search Using an Image
#####################################################

results = mq.index("my-multimodal-index").search(
    # URL of a Hippo
    "https://animalfactguide.com/wp-content/uploads/2013/01/iStock_000003506850XSmall.jpg"
)

print(results)

####################################################
##       Search Using Weights in Queries
####################################################

mq = marqo.Client(url="http://localhost:8882")

try:
    mq.index("my-weighted-query-index").delete()
except:
    pass

mq.create_index("my-weighted-query-index")

mq.index("my-weighted-query-index").add_documents(
    [
        {
            "Title": "Smartphone",
            "Description": "A smartphone is a portable computer device that combines mobile telephone "
            "functions and computing functions into one unit.",
        },
        {
            "Title": "Telephone",
            "Description": "A telephone is a telecommunications device that permits two or more users to"
            "conduct a conversation when they are too far apart to be easily heard directly.",
        },
        {
            "Title": "Thylacine",
            "Description": "The thylacine, also commonly known as the Tasmanian tiger or Tasmanian wolf, "
            "is an extinct carnivorous marsupial."
            "The last known of its species died in 1936.",
        },
    ],
    tensor_fields=["Description"],
)

# Initially we ask for a type of communications device which is popular in the 21st century
query = {
    # A weighting of 1.1 gives this query slightly more importance
    "I need to buy a communications device, what should I get?": 1.1,
    # This will lead to 'Smartphone' being the top result
    "The device should work like an intelligent computer.": 1.3,
}

results = mq.index("my-weighted-query-index").search(q=query)

print("Query 1:")
pprint.pprint(results)

# Now we ask for a type of communications which predates the 21st century
query = {
    # A weighting of 1 gives this query a neutral importance
    "I need to buy a communications device, what should I get?": 1.0,
    # This will lead to 'Telephone' being the top result
    "The device should work like an intelligent computer.": -0.3,
}

results = mq.index("my-weighted-query-index").search(q=query)

print("\nQuery 2:")
pprint.pprint(results)

#####################################################
###       Creating and Searching Indexes 
###      with Multimodal Combination Fields
#####################################################
import marqo
import pprint

mq = marqo.Client(url="http://localhost:8882")

settings = {
    "treat_urls_and_pointers_as_images": True,
    "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
}

try:
    mq.index("my-first-multimodal-index").delete()
except:
    pass

mq.create_index("my-first-multimodal-index", **settings)

mq.index("my-first-multimodal-index").add_documents(
    [
        {
            "Title": "Flying Plane",
            "caption": "An image of a passenger plane flying in front of the moon.",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
        },
        {
            "Title": "Red Bus",
            "caption": "A red double decker London bus traveling to Aldwych",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
        },
        {
            "Title": "Horse Jumping",
            "caption": "A person riding a horse over a jump in a competition.",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
        },
    ],
    # Note that captioned_image must be a tensor field
    tensor_fields=["captioned_image"],
    # Create the mappings, here we define our captioned_image mapping
    # which weights the image more heavily than the caption - these pairs
    # will be represented by a single vector in the index
    mappings={
        "captioned_image": {
            "type": "multimodal_combination",
            "weights": {"caption": 0.3, "image": 0.7},
        }
    },
)

# Search this index with a simple text query
results = mq.index("my-first-multimodal-index").search(
    q="Give me some images of vehicles and modes of transport. I am especially interested in air travel and commercial aeroplanes."
)

print("Query 1:")
pprint.pprint(results)

# Search the index with a query that uses weighted components
results = mq.index("my-first-multimodal-index").search(
    q={
        "What are some vehicles and modes of transport?": 1.0,
        "Aeroplanes and other things that fly": -1.0,
    }
)
print("\nQuery 2:")
pprint.pprint(results)

#####################################################
###                Delete Documents
#####################################################

results = mq.index("my-first-index").delete_documents(
    ids=["article_591", "article_602"]
)

#####################################################
###               Delete Index
#####################################################

results = mq.index("my-first-index").delete()