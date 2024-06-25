# LLM-repo1

## Folders and their purpose
data_source : Contains script to extract floodtags given an AOI

Floodtags_analytics: Contains time series analysis of month long data for FSD_1777 and FSD_1555

models: 
datasets: Contains the two datasets from FSD_1777 and FSD_1555 (json)
language_translation: Contains the translation model mbart50 and the scripts I used to iteratively translate 100 tweets at a time (mbart50 does not support gpu processing)
LLMs: Contains scripts to download LLMs to you conda environment

langchain_implementation:

routerChains_llama_agents : contains the demo presented at the Lunch and learn utilizing agentic workflow
routerChains_llama_OpenAI : 1st demo app utilizing only router chains and Open AI models
routerChains_llama: Demo utilizing only router chains and local LLMs
simple_chain_experiments: Running experiments using wandb and RAGAS for auto LLM eval