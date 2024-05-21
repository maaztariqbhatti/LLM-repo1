from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import time


model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# # translate Hindi to French
# tokenizer.src_lang = "hi_IN"
# encoded_hi = tokenizer(article_hi, return_tensors="pt")
# generated_tokens = model.generate(
#     **encoded_hi,
#     forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
# )
# print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
# # => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."

# # translate Arabic to English
# tokenizer.src_lang = "ar_AR"
# encoded_ar = tokenizer(article_ar, return_tensors="pt")
# generated_tokens = model.generate(
#     **encoded_ar,
#     forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
# )
# tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# # => "The Secretary-General of the United Nations says there is no military solution in Syria."

#Japanese to english
article_ja = "Which locations are receiving flood warnings?"
tokenizer.src_lang = "en_XX"
encoded_hi = tokenizer(article_ja, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["ja_XX"]
).to("cuda")

# your code here    
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
