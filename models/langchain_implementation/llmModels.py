import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

#Llama 2
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
            temperature = 0.5,
            max_new_tokens = 560
        )

    chatModel= HuggingFacePipeline(pipeline=pipeline)

    return chatModel


def loadMistral7b():
    #Empty cuda cache
    torch.cuda.empty_cache()
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    model_m7b = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code = True).to("cuda")
    tokenizer_m7b = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    pipeline_m7b = transformers.pipeline(
            task = "text-generation",
            model = model_m7b,
            return_full_text = True,
            tokenizer = tokenizer_m7b,
            do_sample = True,
            temperature = 0.1,
            max_new_tokens = 560
        )
    chatModel= HuggingFacePipeline(pipeline=pipeline_m7b)

    return chatModel