from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch

device = "cuda" # the device to load the model onto

model_m7b = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto",trust_remote_code = True).to("cuda")
tokenizer_m7b = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

# encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

# model_inputs = encodeds.to(device)
# model.to(device)

# generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
# decoded = tokenizer.batch_decode(generated_ids)

# print(decoded[0])

pipeline_m7b = transformers.pipeline(
        task = "text-generation",
        model = model_m7b,
        return_full_text = True,
        tokenizer = tokenizer_m7b,
        temperature = 0.2,
        max_new_tokens = 560
    )

prompt = "As a data scientist, can you explain what is overfitting?"

sequence = pipeline_m7b(
    prompt,
    do_sample = True,
    top_k = 50,
    top_p = 0.95,
    num_return_sequences = 1,
)

print(sequence[0]["generated_text"])