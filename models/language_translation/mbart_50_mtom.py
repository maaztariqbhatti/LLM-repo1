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
article_ja = "愛知県田原市 極楽\u3000黒川\u3000大草町\u3000ほか ●氾濫開始水位超過 汐川 西野橋 汐川水系汐川 川の防災情報 6月2日(金)8時44分の情報"
tokenizer.src_lang = "ja_XX"
encoded_hi = tokenizer(article_ja, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
).to("cuda")

# your code here    
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
