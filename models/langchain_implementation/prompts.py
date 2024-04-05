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
prompt_template_llama = """<s>[INST] <<SYS>>You are a researcher tasked with answering questions about location extraction. If you dont know the answer then dont respond with false information.
Output answer as list of locations<</SYS>> 
Answer the question based on the following context only: 
{context}
question: {question}
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
prompt_template_mistral = """[INST]You are a researcher tasked with answering questions about location extraction from given context. If you don't know the answer to a question, please don't share false information.
Answer the question based on the following tweet context:
{context}
Question: {question}
Limit the answer to containing location names for examples country, province, city, town, zip code, roads, rivers, seas, prefectures etc.
[/INST]"""
ll = """Output the answer only in JSON in the following format {{"locations : [answer]"}}"""