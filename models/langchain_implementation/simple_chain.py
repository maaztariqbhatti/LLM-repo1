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
import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness
)
import llmModels 
# from ragas.llms import LangchainLLMWrapper
import wandb
import pickle
import sentence_transformers
# 1. Start a W&B Run


# dataPath = "FSD1777_Oct23.json"
dotenv.load_dotenv()
chatModel = llmModels.loadLlamma()

class CustomRetriever(VectorStoreRetriever):
    """Implements re ranking of retriever output using cross encoder"""
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.vectorstore.get_relevant_documents(query=query)

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

        prompt_template  = """Make up 10 dummy tweets as answers to the users question related to flooding events in UK and Scotland. Only return the dummy tweets.
        Question: {question}"""
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
    def predictions_response(self, question, eModel = "bge-large-en-v1.5", rType = "Query", rerank = False,k = 20):
        
        #  LLM initialisation
        model = chatModel

        # Load the data from source
        data = self.json_dataloader()

        #Loading pandas dataframe from picke file
        # with open('/home/mbhatti/mnt/d/LLM-repo1/models/langchain_implementation/fsd_1555_0601_21:59:22_23:59:59.pkl', 'rb') as f:
        #     data = pickle.load(f)

        # Convert to vector store
        vectorstore = self.data_embedding(data, eModel= eModel, rType= rType)

         # Get retriever
        if rerank == True:
            retriever = CustomRetriever(vectorstore=vectorstore.as_retriever(search_kwargs={'k': k}))
        else:
            retriever = vectorstore.as_retriever(search_kwargs={'k': k})

        #Llama template LOCATION    
        default_prompt_template = """<s>[INST] <<SYS>>You are a very smart location entity extracter with good knowledge of all the locations in the world. If you don't know the answer to a question, please don't share false information.
        Your response only contains location names such as country, province, city, town, zip code, roads, rivers, seas.<<SYS>> 
        Answer the Question based only on the following tweet's context only: 
        {context}
        Question: {question}[/INST]"""

        default_prompt = PromptTemplate(template = default_prompt_template, input_variables = ['question', 'context'])
        default_chain = RetrievalQA.from_chain_type(llm = model,
                                chain_type='stuff',
                                retriever=retriever,
                                chain_type_kwargs={"prompt": default_prompt}
                                )
        
        # default_chain.invoke(question)

        # # 1. Start a W&B Run
        config = {"k":[10,12,14,16,18,20,22,24], "LLM":"Mistral-7B-Instruct-v0.2", "Text embedding model": "bge-large-en-v1.5"}
        run = wandb.init(
            project="LLM experiment tracking",
            notes="Llama-2-13b-chat-hf",
            tags=["Flood warning only", "FSD-1777"],
            config=config
        )
        for k in run.config["k"]:
            # print(k)
            # Get retriever
            if rerank == True:
                retriever = CustomRetriever(vectorstore=vectorstore.as_retriever(search_kwargs={'k': k}))
            else:
                retriever = vectorstore.as_retriever(search_kwargs={'k': k})

            # parserStr = StrOutputParser()

            #Open AI template
            # default_prompt_template = """Answer the question based only on the following tweet's context: {context}
            # Question: {question}"""

            #Llama template LOCATION    
            default_prompt_template = """<s>[INST] <<SYS>>You are a very smart location entity extracter with good knowledge of all the locations in the world. If you don't know the answer to a question, please don't share false information.
            Your response only contains location names such as country, province, city, town, zip code, roads, rivers, seas.<<SYS>> 
            Answer the Question based only on the following tweet's context only: 
            {context}
            Question: {question}[/INST]"""

            default_prompt = PromptTemplate(template = default_prompt_template, input_variables = ['question', 'context'])
            default_chain = RetrievalQA.from_chain_type(llm = model,
                                    chain_type='stuff',
                                    retriever=retriever,
                                    chain_type_kwargs={"prompt": default_prompt}
                                    )
            
            #Evaluation using RAGAS
            questions = ["Which places received a flood weather warning?", "How many casualties occured/ people died due to current flooding event?"]

            ground_truths = [["Scotland,Essex,Scotland, Ireland,Haringey,Angus,Dundee way,UK,Perthshire,,Aberdeenshire,Dundee, Stonehaven, A90,Ireland,main road connecting Dundee and Aberdeen, Scotland,Middleton,northern outskirts of Perth,Sheffield,Brechin,River Spen,Alyth"],
                            ["1 death"]]
            answers = []
            contexts = []

            # Inference
            for query in questions:
                answers.append(default_chain.invoke(query)['result'])
                contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

            # To dict
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truths": ground_truths
            }

            # Convert dict to dataset
            dataset = Dataset.from_dict(data)

            result = evaluate(
            dataset = dataset, 
            metrics=[
            context_recall,
            answer_correctness
            ],
            )

            wandb.log({"faithfulness": result['faithfulness'], "answer_relevancy":result['answer_relevancy'],
            "context_recall" : result['context_recall'], "context_relevancy": result['context_relevancy']})

        
        # return result
        


if __name__ == "__main__":

    dataPath = "/home/mbhatti/mnt/d/LLM-repo1/models/langchain_implementation/FSD1777_Oct23.json"
    langChain_analysis = LangChain_analysis(_dataPath = dataPath,
                            _dateFrom = "2023-10-19 21:06:21+00:00",
                            _dateTo = "2023-10-19 23:58:47+00:00")
    prompt = "Which places received a flood weather warning?"

    langChain_analysis.predictions_response(prompt)

    # print(response)
                                                          














