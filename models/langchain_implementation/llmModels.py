import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from torch import cuda, bfloat16

#Llama 2 13b chat
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
            # temperature = 0.1,
            do_sample=False,
            repetition_penalty=1.1
        )

    chatModel= HuggingFacePipeline(pipeline=pipeline)

    return chatModel


def loadMistral7b():
    # #Empty cuda cache
    # torch.cuda.empty_cache()
    # torch.backends.cuda.enable_mem_efficient_sdp(False)
    # torch.backends.cuda.enable_flash_sdp(False)

    model_m7b = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto", trust_remote_code = True).to("cuda")
    tokenizer_m7b = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    model_m7b.generation_config.temperature=None
    model_m7b.generation_config.top_p=None

    pipeline_m7b = transformers.pipeline(
            task = "text-generation",
            model = model_m7b,
            return_full_text = True,
            tokenizer = tokenizer_m7b,
            do_sample=False,
            max_new_tokens = 512
        )
    chatModel= HuggingFacePipeline(pipeline=pipeline_m7b)

    return chatModel

#Llama2-70B-Chat
def loadLlama2_70B():
    model_id = 'meta-llama/Llama-2-70b-chat-hf'

    #Empty cuda cache
    torch.cuda.empty_cache()
    # torch.backends.cuda.enable_mem_efficient_sdp(False)
    # torch.backends.cuda.enable_flash_sdp(False)

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library 35 gb ram 
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # begin initializing HF items, need auth token for these
    hf_auth = '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'


    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
        token=hf_auth
    )

    print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf", token=hf_auth)
    
    model.generation_config.temperature=None
    model.generation_config.top_p=None

    pipeline = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    do_sample=False,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating,
    )

    chatModel= HuggingFacePipeline(pipeline=pipeline)

    return chatModel

#Llama2-70B-Chat
def loadLlama3_8B():

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