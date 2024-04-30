import os
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import transformers
import torch
from pydantic import BaseModel
from typing import Optional
#from accelerate import infer_auto_device_map


#Enter your local directory wehre the model is stored
save_path = "/home/mbhatti/mnt/d/Llama-2-13b-chat-hf"

#Empty cuda cache
torch.cuda.empty_cache()
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


#Call the model and tokenizer from local storage
local_model = AutoModelForCausalLM.from_pretrained(save_path, return_dict=True, trust_remote_code=True, device_map="auto",torch_dtype=torch.bfloat16).to("cuda")
local_tokenizer = AutoTokenizer.from_pretrained(save_path)

#Define the body of the request
class InputData(BaseModel):
    text: str

app = FastAPI()

#Define the status endpoint
@app.get("/status")
def read_root():
    return {"The Llama2-13b-hf API is healthy. Use the /llama13bchathf endpoint to generate inferences. Optional query parms : temp, max_len, top_k"}

#Define the inference endpoint
@app.post("/llama13bchathf")
#Configure inference generation
def get_prediction(input_data: InputData, 
                   temp: Optional[float] = 0.5,
                   max_len: Optional[int] = 1000,
                   top_k: Optional[int] = 5):
    pipeline = transformers.pipeline(
        "text-generation",
        model = local_model,
        tokenizer = local_tokenizer,
        torch_dtype = torch.bfloat16,
    )
    sequences = pipeline(
        input_data.text,
        temperature = temp,
        max_length = max_len,
        do_sample = True,
        top_k = top_k,
        num_return_sequences = 1,
        eos_token_id = local_tokenizer.eos_token_id,
        bos_token_id = local_tokenizer.bos_token_id,
        sep_token_id = local_tokenizer.sep_token_id,
    )
    
    #Display generated text
    return sequences[0]['generated_text'].split("[/INST] ",1)[1]         
