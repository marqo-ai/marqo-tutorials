#####################################################
### STEP 1. Run Marqo
#####################################################

"""
docker pull marqoai/marqo:latest
docker rm -f marqo
docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
"""

#####################################################
### STEP 2. Download LLM Model
#####################################################

# You can download the model directly from: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf?download=true

# Once downloaded, place this into /models/

#####################################################
### STEP 3. Set Up LLM
#####################################################

from llama_cpp import Llama

# Initialize the Llama model
LLM = Llama(
    model_path="starter-guides/rag/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    n_ctx=4096,  # Increased context size for handling larger inputs
    n_gpu_layers=1  # Enable GPU acceleration if available
)

# Define the question to be asked
question = "Who won gold in the women's 100 metre race at the Paris Olympics 2024?"

# Generate the response using the Llama model
response_stream = LLM(
    question, 
    max_tokens=512,  # Maximum number of tokens in the response
    stop=["Q:"],  # Stop specifies a list of stop sequences to end the generation
    stream=True  # The response should be streamed
)

# Capture the output - the response_stream provides a stream of token choices; we join them to form a complete response
response = ''.join([item['choices'][0]['text'] for item in response_stream])

# Extract only the first paragraph from the response - Splits the response by newline and takes the first part
first_paragraph = response.split('\n', 1)[0].strip()

# Print the first paragraph of the response
print("Just LLM Response:", first_paragraph)

#####################################################
### STEP 4. Define Documents to Perform RAG
#####################################################

# Define documents you want to use as part of your RAG
DOCUMENTS = [
    {
        '_id': '1',
        'date': '2024-08-03',
        'website': 'www.bbc.com',
        'Title': "Alfred storms to Olympic 100m gold in Paris.",
        'Description': "Julien Alfred stormed to the women's 100m title at Paris 2024 to make history as St Lucia's first Olympic medallist. As the rain teemed down at a raucous Stade de France, Alfred, 23, dominated the final and sealed victory by a clear margin in a national record 10.72 seconds. American world champion Sha'Carri Richardson took silver in 10.87, with compatriot Melissa Jefferson (10.92) third. Great Britain's Daryll Neita finished four-hundredths of a second off the podium in fourth, crossing the line in 10.96. Neita produced the best finish by a British female athlete in an Olympic sprint final for 64 years but that will be of little consolation in her pursuit of a first individual global medal."
    },
    # Add more documents here
]

#####################################################
### STEP 5. Use Marqo to Perform RAG
#####################################################

from marqo import Client

# Your index name
index_name = 'news-index-api'

# Set up Marqo Client
client = Client(url='http://localhost:8882')

# We create the index. Note if it already exists an error will occur
# as you cannot overwrite an existing index. For this reason, we delete
# any existing index 
try:
    client.index(index_name).delete()
except:
    pass

# Create Marqo index
client.create_index(index_name)

# Indexing documents 
client.index(index_name).add_documents(DOCUMENTS, tensor_fields= ["Title", "Description"])

# Ensure only documents with this date are included
date = '2024-08-03'

# Peform search on Marqo index with the same question as before
results = client.index(index_name).search(
    q=question, 
    filter_string=f"date:{date}", 
    limit=5)

# Print out the results
print(results)

# Use the results obtained by Marqo as part of context to the LLM when asking the question again
context = ''
for i, hit in enumerate(results['hits']):
    title =  hit['Title']
    text = hit['Description']
    context += f'Source {i}) {title} || {" ".join(text.split()[:60])}... \n'

def get_context_prompt(question, context):
    """ LLM prompt with text-based context from Marqo search. """
    return f'Background: \n{context}\n\nQuestion: {question}\n\nAnswer:'
    
# Obtain the prompt with the content from Marqo
prompt_w_context = get_context_prompt(question=question, context=context)
print("Prompt to input into LLM: ", prompt_w_context)

# Generate the response
response_stream = LLM(prompt_w_context, max_tokens=512, stop=["Q:"], stream=True)

# Capture and print the output
response = ''.join([item['choices'][0]['text'] for item in response_stream])
first_paragraph = response.split('\n\n', 1)[0].strip()
print("LLM & Marqo Response:", first_paragraph)

