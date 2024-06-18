#Hyde
prompt_template_HyDE_OpenAI = """Act as a japanese twitter bot. Write 3 dummy tweets in english to answer the user question. Dont include emojis!
    Question: {question}
    """

### LLAMA 2 ### 
#Default chain 
prompt_template_default = """<s>[INST] <<SYS>>You are a helpful, respectful and honest assistant. If you don't know the answer to a question, please don't share false information.<<SYS>> 
Answer the question based only on the following tweet's context only: {context}
Question: {question}[/INST]"""

#Llama template for returning JSON but needs work
prompt_template_llama_1 = """<s>[INST] <<SYS>>You are a researcher tasked with answering questions about location extraction from given context. If you don't know the answer to a question, please don't share false information.
Output answer in JSON using the following format: {{"flooded_locations": extracted locations}}<</SYS>> 
Answer the question based on context only! Which locations received a flood warning?
context: Flood warnings issued in New york, Miami and Houston[/INST]
answer: {{"flooded_locations: : ["New york", "Miami", "Houston"]}}</s>
<s>[INST]Answer the question based on the following context only! {question}
Output answer in JSON using the following format: {{"flooded_locations": extracted locations}}
context: {context}
[/INST]"""

#Llama template basic    
prompt_template_llama_loc = """<s>[INST] <<SYS>>You are a researcher tasked with answering questions about location extraction. If you dont know the answer then dont respond with false information.
Output answer as list of locations only such as country, province, city, town, roads, rivers, seas, prefectures etc.<</SYS>> 
Answer the question based on the context only!
question: {question}
Context: 
{context}
[/INST]
Locations:"""

prompt_template_llama_api = """<s>[INST] <<SYS>>Act as a location extractor and extract all relevant locations with respect to the user question.<</SYS>> 
Answer the question based on the following context only: 
{context}
Question: {question}
[/INST]"""

prompt_template_llama_default = """<s>[INST] <<SYS>>You are a smart chatbot assistant. If you dont know the answer then dont respond with false information.<</SYS>> 
Answer the question based on the following context only: 
{context}
Question: {question}
[/INST]"""

#Llama template LOCATION    
prompt_template_llama_oneshot = """<s>[INST] <<SYS>>You are a smart researcher tasked with answering questions about location extraction. If you dont know the answer then dont respond with false information.
Output the answer in json using the following json format : {{"locations" : ['location 1', 'location 2']}} <</SYS>> 
Answer the question based on the following context only!
context : Finland expected to snow in the next month
Hervanta to be sunny in the coming days
Heavy snowfall in Helsinki
question : Which locations have expected snowfall? [/INST]
[
{{"locations" : ['Finland', 'Helsinki']}}
]
</s>
<s>[INST]
Answer the question based on the following context only! 
context: {context}
question: {question}
Output the answer in json using the following json format : {{"locations" : ['location 1', 'location 2']}}
[/INST]"""

prompt_template_llama_json = """<s>[INST] <<SYS>>You are a smart researcher tasked with answering questions about location extraction. If you dont know the answer then dont respond with false information.
Output the answer in json using the following json format : {{"locations" : ['location 1', 'location 2']}} <</SYS>> 
Answer the question based on the following context only! 
context: {context}
question: {question}
Output the answer only in json using the following json format : {{"locations" : ['location 1', 'location 2']}}
[/INST]"""


###### MISTRAL ####################

#Flooded locations
prompt_template_mistral = """<s>[INST]Act as a location extractor and extract all relevant locations with respect to the user question.
Answer the question based on the following context only: 
{context}
Question: {question}
[/INST]"""

prompt_template_mistral_default = """<s>[INST]You are a smart chatbot assistant. If you dont know the answer then dont respond with false information.
Answer the question based on the following context only: 
{context}
Question: {question}
[/INST]"""

ll = """Output the answer only in JSON in the following format {{"locations : [answer]"}}"""


# Human casualties 
prompt_template_human = """<s>[INST] You are a analyst tasked with answering questions about human casualties such as deaths and injuries. If you dont know the answer then dont respond with false information.
You provide a sum of casualties. Seperate deaths and injuries in your answers where applicable
Answer the question based on the following context only: 
{context}
question: {question}
[/INST]
Casualties:"""

## Geocode location
prompt_template_geocode = """<s>[INST] <<SYS>>You are a researcher tasked with answering questions about location extraction. If you dont know the answer then dont respond with false information.
Output the answer as geographic coordinate in the format : [latitude, longitude]<</SYS>> 
question: {question}
[/INST]
[latitude, longitude]: """


## LLama 3 prompt templates
prompt_template_llama3_loc = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Act as a location extractor and extract all relevant locations with respect to the user question.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Answer the question based on the following context only: 
{context}
Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

prompt_template_llama3_default = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a smart chatbot assistant. If you dont know the answer then dont respond with false information.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Answer the question based on the following context only: 
{context}
question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

