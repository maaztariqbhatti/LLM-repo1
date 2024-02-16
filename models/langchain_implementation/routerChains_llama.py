from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain, LLMRouterChain, MultiPromptChain, HypotheticalDocumentEmbedder, RetrievalQA
import dotenv
from langchain import hub
import langchainhub
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import Optional
import json
import pandas as pd
from Text_preprocessing import Text_preprocessing
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic.v1 import BaseModel
from typing import List, Dict, Any, Mapping
from langchain.globals import set_debug
from langchain.chains.router.base import MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)
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

dataPath = "FSD1777_Oct23.json"



#Llama 2
@st.cache_resource
def loadLlamma():
    #Enter your local directory wehre the model is stored
    save_path = "/home/mbhatti/mnt/d/Llama-2-13b-chat-hf"

    #Empty cuda cache
    torch.cuda.empty_cache()
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)


    #Call the model and tokenizer from local storage
    local_model = AutoModelForCausalLM.from_pretrained(save_path, return_dict=True, trust_remote_code=True, device_map="auto",torch_dtype=torch.bfloat16).to("cuda")
    local_tokenizer = AutoTokenizer.from_pretrained(save_path)


    # pipeline = transformers.pipeline(
    #         task = "text-generation",
    #         model = local_model,
    #         return_full_text = True,
    #         tokenizer = local_tokenizer,
    #         temperature = 0.01,
    #         do_sample = True,
    #         top_k = 5,
    #         num_return_sequences = 1,
    #         max_length = 4000,
    #         eos_token_id = local_tokenizer.eos_token_id
    #     )
    pipeline = transformers.pipeline(
            task = "text-generation",
            model = local_model,
            return_full_text = True,
            tokenizer = local_tokenizer,
            temperature = 0.1,
            max_new_tokens = 560
        )

    chatModel= HuggingFacePipeline(pipeline=pipeline)

    return chatModel

#Open AI 
@st.cache_resource
def loadOpenAI():
    dotenv.load_dotenv()
    chatModelOpenAI = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.5)

    return chatModelOpenAI

#Load the models
# torch.cuda.empty_cache()
chatModel = loadLlamma()
chatModelOpenAI = loadOpenAI()

system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

class DKMultiPromptChain (MultiRouteChain):
    destination_chains: Mapping[str, Chain]
    """Map of name to candidate chains that inputs can be routed to. Not restricted to LLM"""

class Config(BaseModel):
    input_key_map: Optional[Dict]
    destination_chain: Any


class InputAdapterChain(Chain):
    config:Config
    output_key: str = "result"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.
        :meta private:
        """
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """"""
        for k, v in self.config.input_key_map.items():
            if k in inputs.keys():
                inputs[v] = inputs[k]
                del inputs[k]
        # print("##### inputs is now")
        # print(inputs)

        return self.config.destination_chain.invoke(inputs)


class PromptFactory():
    location_template = """<s>[INST] <<SYS>>You are a very smart location entity extracter with good knowledge of all the locations in the world. 
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    Your response only contains location names such as country, province, city, town, zip code, roads, rivers, seas, oceans.<<SYS>>  
    Answer question according to the following context only!
    Context: {context}
    Question:{question}[/INST]"""


    numbers_template = """<s>[INST] <<SYS>>Your are an expert at detecting and extracting numbers and important figures within sentences. Your provide a summary of the gathered information as truthfully as possible.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <<SYS>> 
    Answer question according to the following context only!
    Context: {context}
    Question:{question}[/INST]"""


    prompt_infos = [
        {
            'name': 'casualties_detector',
            'description': 'Good for summarizing number based questions such as number of deaths, injuries, property damages etc',
            'prompt_template': numbers_template
        },
        {
            'name': 'location_extractor',
            'description': 'Good for extracting locations from a context',
            'prompt_template': location_template
        }

    ]

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

        #Re order according to long text re-order (Important context in start and end)
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(sorted_docs)

        #Remove cross encoder score so re ordered context is back to its orignal form
        for doc in reordered_docs:
            doc.metadata.pop("cross-encoder_score")
        return reordered_docs

class LangChain_analysis:
    """
    Utilise the power of LLMS and Langchain to answer questions about flood tags reponse.
    Different data loader functions to load different data storage types
    """
    def __init__(self, _dataPath, _dateFrom, _dateTo):
        
        self.dataPath = _dataPath
        self.dateFrom = _dateFrom
        self.dateTo = _dateTo

    """Load relevant fields of flood tags api json response"""
    def json_dataloader(self):
        # Load json and extract relevant records in pandas df
        with open(self.dataPath, 'r') as json_file:
            response_dict = json.load(json_file)

        # Convert to pandas df    
        pd.set_option('display.max_colwidth', None)
        df = pd.DataFrame(response_dict)
        df['date'] = pd.to_datetime(df['date'])
        df = df.drop(columns=['id','tag_class', 'source', 'lang', 'urls','locations'])

        #Get data between thresholds
        threshold_datetime_lower = pd.to_datetime(self.dateFrom)
        threshold_datetime_upper = pd.to_datetime(self.dateTo)
        df = df[df['date'] >= threshold_datetime_lower]
        df = df[df['date'] <= threshold_datetime_upper]

        #Pre-process
        preprocess = Text_preprocessing(df)
        df = preprocess.preprocess()
        #Covert date to string
        df['date'] = df['date'].astype(str)
        return df
    
    def bgeEmbeddings(self):
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return model
    
    def hydeEmbedder(self,embeddingsModel):
        model = chatModel

        prompt_template  = """<s>[INST] <<SYS>>Act as a twitter bot who is an expert at generating tweets<<SYS>> 
        Develop a few dummy tweets as possible answers to the question given below 
        Question: {question}[/INST]"""
        prompt = PromptTemplate(input_variables=["question"], template= prompt_template)
        llm_chain_hyde  = LLMChain(llm = model, prompt=prompt)

        embeddings = HypotheticalDocumentEmbedder(llm_chain=llm_chain_hyde,
                                                    base_embeddings=embeddingsModel,
                                                    verbose=True)
        return embeddings
    

    def data_embedding(self, data : list, eModel, rType):
        """Vectorize the data using OpenAI embeddings and store in Chroma db"""
        if eModel != "bge-large-en-v1.5":
            embeddings = OpenAIEmbeddings()
        else:
            embeddings = self.bgeEmbeddings()
        
        if (rType == "Hyde"):
            embeddings = self.hydeEmbedder(embeddings)

        documents = []
        loader = DataFrameLoader(data, page_content_column="text")
        documents.extend(loader.load())

        #Create a vector store
        db = Chroma.Chroma.from_documents(documents,embeddings)

        return db
    
    """Langchain and LLM processing code"""
    def predictions_response(self, input_question, eModel = "bge-large-en-v1.5", rType = "Query", k = 20):
        # Load the data from source
        data = self.json_dataloader()

        # Convert to vector store
        vectorstore = self.data_embedding(data, eModel= eModel, rType= rType)
        
        #Get retriever
        if rType == "Cross-Encoder ranking":
            retriever = CustomRetriever(vectorstore=vectorstore.as_retriever(search_kwargs={'k': k}))
        else:
            retriever = vectorstore.as_retriever(search_kwargs={'k': k})

        #  LLM initialisation
        model = chatModel
        parserStr = StrOutputParser()
        #Define all the destination chains 
        prompt_factory = PromptFactory()
        parserStr = StrOutputParser()
        destination_chains = {}
        for p_info in prompt_factory.prompt_infos:
            name = p_info['name']
            prompt_template = p_info['prompt_template']
            prompt = PromptTemplate(template = prompt_template, input_variables=['question', 'context'])

            chain = RetrievalQA.from_chain_type(llm = model,
                                chain_type='stuff',
                                retriever=retriever,
                                chain_type_kwargs={"prompt": prompt}
                                )
            input_key_map = {"input": "query"}
            adapted_chain = InputAdapterChain(config=Config(destination_chain=chain, input_key_map=input_key_map))

            destination_chains[name] = adapted_chain

        #Default chain and prompt
        default_prompt_template = """<s>[INST] <<SYS>>You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<<SYS>> 
        Answer the question based only on the following tweet's context: {context}
        Question: {question}[/INST]"""
        default_prompt = PromptTemplate(template = default_prompt_template, input_variables = ['question', 'context'])
        default_chain = RetrievalQA.from_chain_type(llm = model,
                                chain_type='stuff',
                                retriever=retriever,
                                chain_type_kwargs={"prompt": default_prompt}
                                )

        adapted_default_chain = InputAdapterChain(config=Config(destination_chain=default_chain, input_key_map=input_key_map))
        default_chain = adapted_default_chain

        ## Generate router chain from prompt --
        destinations = [f"{p['name']}: {p['description']}" for p in prompt_factory.prompt_infos]
        destinations_str = '\n'.join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=['input'],
            output_parser=RouterOutputParser()
        )

        router_chain = LLMRouterChain.from_llm(chatModelOpenAI, router_prompt)

        chain = MultiPromptChain(
            router_chain = router_chain,
            destination_chains = destination_chains,
            default_chain = default_chain,
            verbose=True
            # callbacks=[file_ballback_handler]
        )                    

        return chain.invoke(input_question)
    
    
if __name__ == "__main__":

    # langChain_analysis = LangChain_analysis(_dataPath = "G:/My Drive/LLM-repo1/Floodtags_analytics/FSD1777_Oct23.json",
    #                                 _dateFrom = "2023-10-19 21:06:21+00:00",
    #                                 _dateTo = "2023-10-19 23:58:47+00:00")

    langChain_analysis = LangChain_analysis(_dataPath = dataPath,
                                _dateFrom = "2023-10-19 21:06:21+00:00",
                                _dateTo = "2023-10-19 23:58:47+00:00")

    # question = "Which places/location's recieved a flood warning or evacuation orders? Which places are affected by floods"
    # output = langChain_analysis.predictions_response(question)
    # prompt = "Which places/location's recieved a flood warning or evacuation orders?"
    # response = langChain_analysis.predictions_response(prompt, "bge-large-en-v1.5", "Cross-Encoder ranking")['result']
    # print(response)

    #Streamlit
    st.title("SNS early flood warning 🤖")

    #Side bar to select parameters
    with st.sidebar:
        st.write("Embedding option")

        #Select embedding model
        k = st.number_input("Select number of documents to add inside LLM prompt", min_value= 2, max_value= 35,step=1)

        eModel = st.selectbox(
        "Embedding model",
        ("OpenAI", "bge-large-en-v1.5"),
        index=None, 
        placeholder="Select embedding model...",
        )
        st.write('You selected:', eModel)
        #select retrieval type
        rType= st.selectbox(
        "Retrieval type",
        ("Query", "Hyde", "Cross-Encoder ranking"),
        index=None, 
        placeholder="Select retrieval type...",
        )
        st.write('You selected:', rType)

    #Predefined prompts
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        floodLoc = st.button("Find flooded locations")
    with col2:
        roadsClosure = st.button("Roads/transport closure")

    st_input = st.chat_input("Talk to me")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    if floodLoc == True:
        hard_prompt = "Which places/location's received a flood warning or evacuation orders?"
        st_input = hard_prompt
    if roadsClosure == True:
        st_input = "Is there a mention of closure of roads? Closure of transport services? If so where"

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
                response = langChain_analysis.predictions_response(prompt, eModel, rType, k)['result']
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})














