import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, LLMRouterChain, MultiPromptChain, HypotheticalDocumentEmbedder, RetrievalQA
import dotenv
from langchain_core.prompts import PromptTemplate
from typing import Optional
import json
import pandas as pd
from Text_preprocessing import Text_preprocessing
from langchain_community.document_loaders import DataFrameLoader
from typing import List, Dict, Any, Mapping
from langchain.globals import set_debug

from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

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
    answer_correctness,
    answer_similarity,
    context_entity_recall
)
import llmModels 
import wandb
import pickle
import prompts
import groundTruths

from langsmith import Client
from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate
import openai
# from langsmith.wrappers import wrap_openai


dotenv.load_dotenv()

### PARAMETER CHECKPOINT ####

#Open AI
chatModelAI = ChatOpenAI()

# # # # Llama 2 13 B chat
# chatModel_llama13b = llmModels.loadLlamma()

# # # Mistral 7B chat
# chatModel_mistral7b = llmModels.loadMistral7b()

# #70B
# chatModel_llama70b = llmModels.loadLlama2_70B()

## Llama 3 8B
chatModel_llama3_8B = llmModels.loadLlama3_8B()

##FSD_1777
dataPath = "/home/mbhatti/mnt/d/LLM-repo1/models/langchain_implementation/FSD1777_Oct23.json"
dateFrom = "2023-10-19T09:00:00+00:00" #2023-10-19T18:58:41Z for 200 tweets
dateTo = "2023-10-19T18:50:00+00:00"

##FSD_1555
# dataPath = "/home/mbhatti/mnt/d/LLM-repo1/models/langchain_implementation/fsd_1555_0601_21_59_22TO23_59_59.pkl"
# dateFrom = "2023-06-01 22:59:45+00:00" 
# dateTo = "2023-06-01 23:59:59+00:00" #200 tweets labelled

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

        # #Remove 10 worst performing docs from the list 
        sorted_docs = sorted_docs[:-5]

        #Re order according to long text re-order (Important context in start and end)
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(sorted_docs)

        #Remove cross encoder score so re ordered context is back to its orignal form
        for doc in reordered_docs:
            doc.metadata.pop("cross-encoder_score")
        return reordered_docs

"""Load relevant fields of flood tags api json response"""
def json_dataloader(dataPath = dataPath, dateFrom = dateFrom, dateTo = dateTo):
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

"""Load relevant fields of flood tags api json response"""
def dataframe_dataloader(dataPath = dataPath, dateFrom = dateFrom, dateTo = dateTo):
    # Loading pandas dataframe from picke file
    with open(dataPath, 'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    # df = df.drop(columns=['id','tag_class', 'source', 'lang', 'urls','locations'])

    #Get data between thresholds
    threshold_datetime_lower = pd.to_datetime(dateFrom)
    threshold_datetime_upper = pd.to_datetime(dateTo)
    df = df[df['date'] >= threshold_datetime_lower]
    df = df[df['date'] <= threshold_datetime_upper]

    #Remove duplicates
    df  = df.drop_duplicates(subset=["text"], keep=False)
    
    #Covert date to string
    df['date'] = df['date'].astype(str)
    return df


def hydeEmbedder(embeddingsModel):
    
    model = chatModelAI
    prompt_template  = prompts.prompt_template_HyDE_OpenAI
    prompt = PromptTemplate(input_variables=["question"], template= prompt_template)
    llm_chain_hyde  = LLMChain(llm = model, prompt=prompt)

    embeddings = HypotheticalDocumentEmbedder(llm_chain=llm_chain_hyde,
                                                base_embeddings=embeddingsModel,
                                                verbose=True)
    return embeddings


def data_embedding(data : list, eModel = "bge-large-en-v1.5", rType = "Query"):
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
    print(len(db._collection.get()['ids']))
    return db


"""For running a single query"""
def predictions_response(question, eModel = "bge-large-en-v1.5", rType = "Query", rerank = False,k = 3):
    
    #  LLM initialisation
    model = chatModel_llama3_8B

    # Load the data from source
    data = json_dataloader()

    # Loading pandas dataframe from picke file - FSD-1555
    # data = dataframe_dataloader()

    # Convert to vector store
    vectorstore = data_embedding(data, eModel= eModel, rType= rType)

        # Get retriever
    if rerank == True:
        retriever = CustomRetriever(vectorstore=vectorstore.as_retriever(search_kwargs={'k': k+5}))
    else:
        retriever = vectorstore.as_retriever(search_kwargs={'k': k})


    default_prompt = PromptTemplate(template = prompts.prompt_template_llama3_loc, input_variables = ['question', 'context'])
    default_chain = RetrievalQA.from_chain_type(llm = model,
                            chain_type='stuff',
                            retriever=retriever,
                            chain_type_kwargs={"prompt": default_prompt},
                            return_source_documents = True
                            )
    
    return default_chain.invoke(question)

def langsmith_eval():

    client = Client()
    datasetId = "5a50d25a-fa12-42c5-97bc-778f7ce96eaa"
    # Define dataset: these are your test cases
    dataset_name = "FSD_1777_preFloodpeak_evac"
    # dataset = client.create_dataset(dataset_name, description="Evacuation orders for fsd_1777 pre flood peak.")
    
    # client.share_run
    client.create_examples(
        inputs=[
            {"question": "Which locations are specifically receiving evacuation orders?"},
        ],
        outputs=[
            {"must_mention": ["evacuation"]}
        ],
        dataset_id=datasetId,
    )
    
    # # Define AI system
    # openai_client = wrap_openai(openai.Client())
    def predict(inputs: dict) -> dict:
        response  = predictions_response(inputs["question"])
        return {"output": response['result']}

    # Define evaluators
    def must_mention(run: Run, example: Example) -> dict:
        prediction = run.outputs.get("output") or ""
        required = example.outputs.get("must_mention") or []
        score = all(phrase in prediction for phrase in required)
        for phrase in required:
            print(phrase)
        return {"key":"must_mention", "score": score}
    
    def f1_score_summary_evaluator(runs: List[Run], examples: List[Example]) -> dict:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for run, example in zip(runs, examples):
            # Matches the output format of your dataset
            reference = example.outputs["answer"]
            # Matches the output dict in `predict` function below
            prediction = run.outputs["prediction"]
            if reference and prediction == reference:
                true_positives += 1
            elif prediction and not reference:
                false_positives += 1
            elif not prediction and reference:
                false_negatives += 1
        if true_positives == 0:
            return {"key": "f1_score", "score": 0.0}

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return {"key": "f1_score", "score": f1_score}

    experiment_results = evaluate(
        predict, # Your AI system
        data=dataset_name, # The data to predict and grade over
        evaluators=[must_mention], # The evaluators to score the results
        experiment_prefix="evac_mentions", # A prefix for your experiment names to easily identify them
        metadata={
        "version": "1.0.0",
        },
    )

if __name__ == "__main__":

    # prompt = "Which locations are specifically receving evacutaion orders?"
    # predictions_response(prompt)

    langsmith_eval()

    # run_sweep()

                                                    














