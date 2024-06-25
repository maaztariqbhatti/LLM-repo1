import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, RetrievalQA
import dotenv
from langchain_core.prompts import  PromptTemplate
from typing import Optional
import json
import pandas as pd
from Text_preprocessing import Text_preprocessing
from langchain_community.document_loaders import DataFrameLoader
from typing import List
from langchain_community.vectorstores import chroma as Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic.v1 import Field
from langchain_core.documents import Document
from langchain_community.document_transformers import (
    LongContextReorder
)
from sentence_transformers import CrossEncoder
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chains import LLMChain
import folium
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
import ast

dataPath = "FSD1777_Oct23.json"
print(os.getcwd())

#The decorator caches the model once loaded onto GPU memory
@st.cache_resource
#Llama3-8B-Chat
def loadLlamma():

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto").to("cuda")

    model.generation_config.temperature=None
    model.generation_config.top_p=None

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    pipeline = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    eos_token_id=terminators,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    do_sample=False, # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512  # mex number of tokens to generate in the output
    )

    chatModel= HuggingFacePipeline(pipeline=pipeline)

    return chatModel

@st.cache_resource
def loadMistral7b():
    model_m7b = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code = True).to("cuda")
    tokenizer_m7b = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    pipeline_m7b = transformers.pipeline(
            task = "text-generation",
            model = model_m7b,
            return_full_text = True,
            tokenizer = tokenizer_m7b,
            do_sample = True,
            temperature = 0.1,
            max_new_tokens = 512
        )
    chatModel= HuggingFacePipeline(pipeline=pipeline_m7b)

    return chatModel

#Open AI 
@st.cache_resource
def loadOpenAI():
    dotenv.load_dotenv()
    chatModelOpenAI = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.0)
    return chatModelOpenAI

#Open AI 
@st.cache_resource
def loadOpenAI_gpt4o():
    dotenv.load_dotenv()
    chatModelOpenAI4o = ChatOpenAI(model="gpt-4o", temperature=0.0)
    return chatModelOpenAI4o

#Load the models
dotenv.load_dotenv()
llm = loadLlamma()
chatModelAI = ChatOpenAI()
llmGPT40 = loadOpenAI_gpt4o()

# chatMistral = loadMistral7b()



class CustomRetriever(VectorStoreRetriever):
    """Implements re ranking of retriever output using cross encoder"""
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.vectorstore.get_relevant_documents(query=query)
        # return sorted(results, key=lambda doc: doc.metadata.get('source'))

        # Get unique documents retrieved (if multiple queries used with hyde)

        # unique_contents = set()
        # unique_docs = []
        # for sublist in docs:
        #     for doc in sublist:
        #         if doc[1] not in unique_contents:
        #             unique_docs.append(doc)
        #             unique_contents.add(doc[1])
        # unique_contents = list(unique_contents)
        # print(unique_contents)
        
        # Cross encoder re ranking 
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        pairs = []
        for doc in docs:
            pairs.append([query, doc.page_content])
        scores = cross_encoder.predict(pairs)

        i = 0
        #Add scores to document
        for doc in docs:
            doc.metadata["cross-encoder_score"] = scores[i]
            i = i+1
        #Sort documents according to score
        sorted_docs = sorted(docs, key=lambda doc: doc.metadata.get('cross-encoder_score'), reverse=True)

        # #Remove 5 worst performing docs from the list 
        sorted_docs = sorted_docs[:-5]

        #Re order according to long text re-order (Important context in start and end)
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(sorted_docs)

        #Remove cross encoder score so re ordered context is back to its orignal form
        for doc in reordered_docs:
            doc.metadata.pop("cross-encoder_score")
        return reordered_docs
    

def json_dataloader(dataPath, dateFrom, dateTo):
    # Load json and extract relevant records in pandas df
    with open(dataPath, 'r') as json_file:
        response_dict = json.load(json_file)

    # Convert to pandas df    
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(response_dict)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(columns=['id','tag_class', 'source', 'lang', 'urls','locations'])

    #Get data between thresholds
    threshold_datetime_lower = pd.to_datetime(dateFrom)
    threshold_datetime_upper = pd.to_datetime(dateTo)
    df = df[df['date'] >= threshold_datetime_lower]
    df = df[df['date'] <= threshold_datetime_upper]

    #Remove duplicates
    df  = df.drop_duplicates(subset=["text"], keep=False)

    #Pre-process
    preprocess = Text_preprocessing(df)
    df = preprocess.preprocess()
    #Covert date to string
    df['date'] = df['date'].astype(str)
    return df

def bgeEmbeddings():
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return model



def data_embedding(data : list, eModel, rType):
    """Vectorize the data using OpenAI embeddings and store in Chroma db"""
    if eModel != "bge-large-en-v1.5":
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = bgeEmbeddings()
    
    if (rType == "Hyde"):
        embeddings = hydeEmbedder(embeddings)

    documents = []
    loader = DataFrameLoader(data, page_content_column="text")
    documents.extend(loader.load())

    #Change this -- removal of duplicates
    db = Chroma.Chroma.from_documents(documents,embeddings)
    if db._client.list_collections() != None:
        for collection in db._client.list_collections():
            ids = collection.get()['ids']
            print('REMOVE %s document(s) from %s collection' % (str(len(ids)), collection.name))
            if len(ids): collection.delete(ids)

    #Create a vector store
    db = Chroma.Chroma.from_documents(documents,embeddings)

    return db



def predictions_response(input_question, vectorstore, conversational_memory,rerank = False,k = 20):
    
    #Get retriever
    if rerank == True:
        retriever = CustomRetriever(vectorstore=vectorstore.as_retriever(search_kwargs={'k': k+5}))
    else:
        retriever = vectorstore.as_retriever(search_kwargs={'k': k})
    
    # Prompt and chain for Twitter DB-----------------
    prompt_template_llama3_loc = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a smart chatbot assistant. If you dont know the answer then dont respond with false information.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Answer the question based on the following context only: 
    {context}
    Question: {question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    default_prompt = PromptTemplate(template = prompt_template_llama3_loc, input_variables = ['question', 'context'])

    # default_prompt.format(sysprompt = sys_prompt)
    # retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": default_prompt},
        verbose = True,
        return_source_documents=True
    )


    # #Prompt and chain for extracting geolocations --------------
    # from langchain.chains import LLMChain
    # from langchain import PromptTemplate
    # from langchain.chains import LLMChain

    # llm = ChatOpenAI(model_name='gpt-3.5-turbo-1106')

    geoLocTemplate = """
    Act as geo locator. Extract the geopoint coordinates according to the question in the following json format: 
    {{'location':'Location name', 
    'latitude' : 12.2,
    'longitude' : 2.33
    }}

    question: {question}
    """

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=geoLocTemplate,
    )
    # description = "It is software dev firm specifically focusing on automation software"
    # prompt_template.format(firm_description=description)
    llmGPT40 = ChatOpenAI(
        model_name='gpt-4o',
        temperature=0.0
    )

    chaingeo = LLMChain(llm=llmGPT40, prompt=prompt_template)   

    tools = [
        Tool(
            name='Twitter database',
            func=qa.invoke,
            description=(
                'Use this tool to answer flooding related questions'
            )
        ),
        Tool(
        name='Geo location extraction',
        func=chaingeo.run,
        description=(
            'Use this tool to extract geolocation coordinates'
        )
        )
    ]

    from langchain.agents import initialize_agent

    # conv
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=chatModelAI,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory,
        return_source_documents=True,
        return_intermediate_steps=True
    )
            

    # query = "Any deaths reported due to flooding? "
    results = agent(input_question)
    return [results, conversational_memory]
    
if __name__ == "__main__":

    #Streamlit
    st.title("SNS early flood warning 🤖")
    response ={}

    #Side bar to select parameters
    with st.sidebar:
        
        AOI = st.text_input(label="Area of interest", value="[129.438857, 30.100228, 141.185201, 37.513876]", disabled=True)

        #From and to date time 
        st.write("Select duration of event")
        start_date = st.text_input(label='Start date')
        end_date = st.text_input(label='End date')

        #Select the model 
        llm_model = st.selectbox(
        "Select LLM",
        ("Llama3-8bChat", "Mistral7bInstruct"),
        index=None, 
        placeholder="Select LLM...",
        )

        st.write("Embedding option")
        #Select embedding model
        k = st.number_input("Select number of documents to add inside LLM prompt context", min_value= 2, max_value= 50,step=1)

        eModel = st.selectbox(
        "Embedding model",
        ("OpenAI", "bge-large-en-v1.5"),
        index=None, 
        placeholder="Select embedding model...",
        )

        #select retrieval type
        rType= st.selectbox(
        "Retrieval type",
        ("Query", "Hyde"),
        index=None, 
        placeholder="Select retrieval type...",
        )

        #select weather to use cross encoder or not
        rerank= st.checkbox("Use cross-encoder re-rank?")
        if rerank:
            st.write("The retrieved documents will be re ranked!")

        #Load data
        loadData = st.button("Load data")
        if (loadData == True):
            with st.spinner():
                data = json_dataloader(dataPath, start_date, end_date)
                # Convert to vector store
                vectorstore = data_embedding(data, eModel= eModel, rType= rType)
                st.session_state.vectorstore = vectorstore

                # Initialize conversational memory
                conversational_memory = ConversationBufferWindowMemory(
                    memory_key='chat_history',
                    k=1,
                    return_messages=True,
                    output_key='output'
                )
                st.session_state.conversational_memory = conversational_memory
            
    choice = st.radio(label='', options=["Chat", "Data Review", "Map"], horizontal=True)

    # with SNSbot:
    if choice == "Chat":
        #Predefined prompts
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            floodLoc = st.button("Flood warnings")
        with col2:
            roadsClosure = st.button("Find roads closure")
        with col3:
            evacuation = st.button("Evacuation orders")
        with col4:
            casualties = st.button("Human casualties")
        

        st_input = st.chat_input("Talk to me")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
            
        if floodLoc == True:
            hard_prompt = "Which locations are receiving flood warnings?"
            st_input = hard_prompt
        if roadsClosure == True:
            st_input = "Is there mention of closure of roads? If yes which roads/highways are shut down due to flooding?"
        if evacuation == True:
            st_input = "Which locations have received evacuation orders?"
        if casualties == True:
            st_input = "Mention of deaths due to storm babet?"

        # React to user input
        if prompt := st_input:
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
                # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                with st.spinner():
                    #Chatbot response
                    response, conversational_memory =  predictions_response(prompt, st.session_state.vectorstore, st.session_state.conversational_memory,rerank, k)
                    st.session_state.conversational_memory = conversational_memory
                    #Store in memory for other tabs
                    st.session_state.response = response

                    #FOR TWITTER DATABASE
                    #Get response
                    if response['intermediate_steps'][0][0].tool == "Twitter database":
                        output = response['intermediate_steps'][0][1]['result']
                        # st.markdown(output)

                    # FOR GEO LOCATION
                    # Get response
                    elif response['intermediate_steps'][0][0].tool == "Geo location extraction":
                        output = response['intermediate_steps'][0][1]
                        # st.markdown(output)

                    else:
                        output = response['result'] 
                        # st.markdown(output)
                
                    st.markdown(output)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": output})
            

    if choice == 'Data Review':
        #Get source docs
        if st.session_state.response['intermediate_steps'][0][0].tool == "Twitter database":
        # if response['intermediate_steps'][0][0].tool == "Twitter database":
            response1 = st.session_state.response['intermediate_steps'][0][1]
            pd.set_option('display.max_colwidth', None)
            tweets_df = pd.DataFrame([docs.page_content for docs in response1['source_documents']], columns=["Tweets"])
            st.dataframe(tweets_df)

    if choice == 'Map':
        
        if st.session_state.response['intermediate_steps'][0][0].tool == "Geo location extraction":
            # Initialize a folium map centered around the UK
            # m = folium.Map(location=[55.3781, -3.4360], zoom_start=6)


            data = st.session_state.response['intermediate_steps'][0][1]

            if '[' in data: 
                formatted_data = data[data.find('['):data.find(']')]
                formatted_data = formatted_data + "]"
                print("\n")
                print(type(formatted_data))
                print("\n")

                x = ast.literal_eval(formatted_data)
                # # x = json.loads(formatted_data)
                print(type(x))
                df = pd.DataFrame(x)

            else:
                formatted_data = data[data.find('{'):data.find('}')]
                formatted_data = formatted_data + "}"
                print("\n")
                print(type(formatted_data))
                print("\n")

                x = ast.literal_eval(formatted_data)
                # # x = json.loads(formatted_data)
                print(type(x))
                print(x)
                df = pd.DataFrame(x, index=[0])
            

            # Convert latitude and longitude to geometry points
            geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]

            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometry)

            # Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
            gdf.set_crs(epsg=4326, inplace=True)

            # Initialize a folium map centered around the UK
            m = folium.Map(location=[55.3781, -3.4360], zoom_start=6)

            # Add points to the map
            for idx, row in gdf.iterrows():
                folium.Marker([row['latitude'], row['longitude']], popup=row['location']).add_to(m)

            # # call to render Folium map in Streamlit
            st_data = st_folium(m, width = 1000)










