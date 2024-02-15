from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import dotenv
from langchain import hub
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from typing import Optional
import json
import pandas as pd
from Text_preprocessing import Text_preprocessing
from langchain_community.document_loaders import DataFrameLoader
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.base import Chain
from langchain_community.vectorstores import chroma as Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from sentence_transformers import CrossEncoder
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.retriever import BaseRetriever
from typing import List
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic.v1 import Field
from langchain_core.documents import Document
from langchain_community.document_transformers import (
    LongContextReorder
)
dataPath = "FSD1777_Oct23.json"

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
        dotenv.load_dotenv()
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
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        return model
    
    def hydeEmbedder(self,embeddingsModel):
        model = ChatOpenAI()

        prompt_template  = """Please answer the users question related to flooding events in UK and Scotland
        Question: {question}
        Answer:"""
        prompt = PromptTemplate(input_variables=["question"], template= prompt_template)
        llm_chain_hyde  = LLMChain(llm = model, prompt=prompt)

        embeddings = HypotheticalDocumentEmbedder(llm_chain=llm_chain_hyde,
                                                    base_embeddings=embeddingsModel,
                                                    verbose=True)
        return embeddings
    
    def data_embedding(self, data : list):
        """Vectorize the data and store in Chroma db"""
        embeddings = OpenAIEmbeddings()

        #BGE - embedding model
        # embeddings = self.bgeEmbeddings()

        #HyDE embeddings
        
        # embeddingModel = self.bgeEmbeddings()

        # embeddings = self.hydeEmbedder(embeddingModel)

        documents = []
        loader = DataFrameLoader(data, page_content_column="text")
        documents.extend(loader.load())

        #Create a vector store
        db = Chroma.Chroma.from_documents(documents,embeddings)

        # # query the DB
        # query = "Which locations are mentioned to be flooded?"
        # docs = db.similarity_search(query, k =10)
        # # print results
        # print(docs)

        return db
    
    """Langchain and LLM processing code"""
    def predictions_response(self):
        # Load the data from source
        data = self.json_dataloader()

        # Convert to vector store
        vectorstore = self.data_embedding(data)

        # Define retriever from store
        #mmr
        #retriever=vectorstore.as_retriever(search_type = "mmr",search_kwargs={'k': 15, 'fetch_k': 50})
        #cosine similarity
        # retriever=vectorstore.as_retriever(search_kwargs={'k': 10})
        location_ext = "Return a list of flooded locations and areas currently susceptible to flooding, encompassing various geographic scales including countries, states, cities, towns, roads, and buildings."
        event_name = "Extract the names of all ongoing natural disasters including but not limited to storms, hurricanes, earthquakes, floods, tornadoes, and tsunamis"
        query = "Which places/location's recieved a flood warning or evacuation orders? Which places are affected by floods"
        custom_retriever = CustomRetriever(vectorstore=vectorstore.as_retriever(search_kwargs={'k': 10}))

        docs = custom_retriever.get_relevant_documents(query=query)
        print(docs)


        
if __name__ == "__main__":
      
    langChain_analysis = LangChain_analysis(_dataPath = dataPath,
                                    _dateFrom = "2023-10-19 21:06:21+00:00",
                                    _dateTo = "2023-10-19 23:58:47+00:00")
    
    langChain_analysis.predictions_response()












