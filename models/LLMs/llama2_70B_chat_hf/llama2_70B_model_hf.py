from torch import cuda, bfloat16
import transformers 
import torch
# BNB_CUDA_VERSION=
# import bitsandbytes
model_id = 'meta-llama/Llama-2-70b-chat-hf'

#Empty cuda cache
torch.cuda.empty_cache()
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(f"Model loaded on {device}")

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
# model_config = transformers.AutoConfig.from_pretrained(
#     model_id,
#     token=token
# )

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth
)
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf", token=hf_auth)
# model.eval()
