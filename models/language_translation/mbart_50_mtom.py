from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import time

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

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
article_ja = "静岡県御前崎市 6月9日（金） \n【警戒レベル4相当】静岡県御前崎市内で土砂災害と河川洪水の危険が上昇 危険な区域を地図で確認 #緊急速報 #静岡県 #御前崎市 #土砂災害"
tokenizer.src_lang = "ja_XX"
encoded_hi = tokenizer(article_ja, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
start = time.process_time()
# your code here    
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
print(time.process_time() - start)