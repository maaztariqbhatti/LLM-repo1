from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import dotenv

dotenv.load_dotenv()
hf_auth = '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'
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto").to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    # {"role": "system", "content": "Act as a location extractor and extract all relevant locations with respect to the user question."},

    {"role": "user", "content": """Does the following tweet specifically mention evacuation orders:
     More than 400 properties in Brechin in Angus – and in the villages of Tannadice and Finavon – have been told to leave their homes with flood defences expected to be breached within 24 hours.
    """},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

model.generation_config.temperature=None
model.generation_config.top_p=None

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=False
    # temperature=0.01
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))