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
    {"role": "system", "content": "You are a smart chat bot that only returns json output and nothing else"},
    {"role": "user", "content": """
    Question: From the following context extract all the location entities in the following json format : {{'flood_warning_locations' : ['location 1', 'location 2']}}

     Context: According to the provided context, the following locations are receiving flood warnings:

    1. Angus Glens region in Scotland
    2. Barbourne in Worcester, England
    3. The River Maun near Retford, England
    4. The coastlines at Bridlington and Scarborough in Yorkshire, England
    5. The banks of the Tyne at North and South Shields in Newcastle, England
    6. Brechin, Angus, Scotland (red alert and severe flood warning)
    7. North Sea at North and South Shields (North Tyneside, South Tyneside)
    8. River South Esk area, Angus, Scotland (severe flood warning)
    9. North Wales (flood risk from heavy rain on Friday and early Saturday)
    10. West and North Yorkshire (possible localised flooding)
    11. Findhorn, Nairn, Moray, and Speyside region (localised flood warning)

    Please note that this information is based on the provided context and may not be exhaustive or up-to-date. It is essential to check for the latest updates and follow official guidance from authorities for the most accurate and reliable information.
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