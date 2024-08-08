#####################################################
### STEP 1. Setting UP
#####################################################

# 1. Sign Up to Marqo Cloud: https://cloud.marqo.ai/
# 2. Get a Marqo API Key: https://www.marqo.ai/blog/finding-my-marqo-api-key


#####################################################
### STEP 2. Import and Define any Helper Functions
#####################################################

from marqo import Client
import json
import math
import numpy as np
import copy
import pprint

def read_json(filename: str) -> dict:
    """
    Reads a JSON file and returns its content as a dictionary.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file as a dictionary.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def clean_data(data: dict) -> dict:
    """
    Cleans the data by removing '- Wikipedia' from the title and converting docDate to a string.

    Args:
        data (dict): The input data dictionary with keys 'title' and 'docDate'.

    Returns:
        dict: The cleaned data dictionary.
    """
    data['title'] = data['title'].replace('- Wikipedia', '')
    data["docDate"] = str(data["docDate"])
    return data


def split_big_docs(data, field='content', char_len=5e4):
    """
    Splits large documents into smaller chunks based on a specified character length.

    Args:
        data (list): A list of dictionaries, each containing a 'content' field or specified field.
        field (str, optional): The field name to check for length. Default is 'content'.
        char_len (float, optional): The maximum character length for each chunk. Default is 5e4.

    Returns:
        list: A list of dictionaries, each containing a chunked version of the original content.
    """
    new_data = []
    for dat in data:
        content = dat[field]
        N = len(content)

        if N >= char_len:
            n_chunks = math.ceil(N / char_len)
            new_content = np.array_split(list(content), n_chunks)

            for _content in new_content:
                new_dat = copy.deepcopy(dat)
                new_dat[field] = ''.join(_content)
                new_data.append(new_dat)
        else:
            new_data.append(dat)
    return new_data

#####################################################
### STEP 3. Load the Data
#####################################################

# Load dataset file 
# Change this to where your 'simplewiki.json' is located
dataset_file = "./starter-guides/text-search/simplewiki.json"

# Get the data
data = read_json(dataset_file)
# Clean up the title
data = [clean_data(d) for d in data]
data = split_big_docs(data)

# Take the first 100 entries of the dataset
N = 100 # Number of entries of the dataset
subset_data = data[:N]

print(f"loaded data with {len(data)} entries")
print(f"creating subset with {len(subset_data)} entries")

#####################################################
### STEP 4. Index Some Data with Marqo
#####################################################

# Replace this with your API Key
api_key = "your_api_key"

# Name your index
index_name = 'text-search-cloud'

# Set up the Client
client = Client(
    "https://api.marqo.ai", 
    api_key=api_key
)

# We create the index. Note if it already exists an error will occur
# as you cannot overwrite an existing index. For this reason, we delete
# any existing index 
try:
    client.delete_index(index_name)
except:
    pass

# Create index
client.create_index(
    index_name, 
    model='hf/all_datasets_v4_MiniLM-L6'
)

# Add the subset of data to the index
responses = client.index(index_name).add_documents(
    subset_data, 
    client_batch_size=50,
    tensor_fields=["title", "content"]
)

# Optionally take a look at the responses
# pprint.pprint(responses)

#####################################################
### STEP 5. Search with Marqo
#####################################################

# Create a query 
query = 'what is air made of?'

# Obtain results for this query from the Marqo index 
results = client.index(index_name).search(query)

# We can check the results - let's look at the top hit
pprint.pprint(results['hits'][0])

# We also get highlighting which tells us why this article was returned
pprint.pprint(results['hits'][0]['_highlights'])

# We use lexical search instead of tensor search
results = client.index(index_name).search(query, search_method='LEXICAL')

# We can check the lexical results - lets look at the top hit
pprint.pprint(results['hits'][0])
