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

    {"role": "user", "content": """According to the following context which locations are receiving flood warnings:
     【近畿地方 気象情報 2023年06月01日 15:32】 近畿地方では、1日夜のはじめ頃から3日午前中にかけて、局地的に雷を伴った激しい雨や非常に激しい雨が降る見込みです。2日昼前から夜遅くにかけて、土砂災害や低い土地の浸水、河川の増水や氾濫に警戒してください。
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