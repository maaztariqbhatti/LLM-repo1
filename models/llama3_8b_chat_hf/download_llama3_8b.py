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
    {"role": "system", "content": "You are a smart chat bot."},
    {"role": "user", "content": """Do the following tweets specificaly mention evacuation orders:
    Alyth in Scotland now starting to see some flooding. Much more rain is yet to fall &amp; filter down into these areas. Red warnings are in force
    A fair bit in the north east I’m in no danger of flooding (high ground) but a seaside town down the road a bit is being battered. Look up Stonehaven.
    Please don’t be stuck if you are affected by flooding in Aberdeenshire. The council have set up rest centres for anyone needing refuge. #StormBabet #AberdeenshireCouncil #Stonehaven"""},
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

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=False
    # temperature=0.01
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
