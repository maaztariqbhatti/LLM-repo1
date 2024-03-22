from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, BitsAndBytesConfig
import json
import time
import pandas as pd
import sys
import torch
import pickle
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print(device)
#Empty cuda cache

#Add path to directory
sys.path.append('/home/mbhatti/mnt/d/LLM-repo1/models')
from langchain_implementation import Text_preprocessing

#Parameters for records to translate
dataPath = "/home/mbhatti/mnt/d/LLM-repo1/models/datasets/FSD1555_June23.json" 
dateFrom = "2023-06-01 19:00:00+00:00"
dateTo = "2023-06-01 23:59:59+00:00"
 
"""Load relevant fields of flood tags api json response"""
def json_dataloader_toPandas(dataPath, dateFrom, dateTo):
    # Load json and extract relevant records in pandas df
    with open(dataPath, 'r') as json_file:
        response_dict = json.load(json_file)

    # Convert to pandas df    
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(response_dict)

    #Preprocessing for FSD-1555
    df = df.drop(8862)

    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(columns=['id','tag_class', 'source', 'lang', 'urls','locations'])

    #Get data between thresholds
    threshold_datetime_lower = pd.to_datetime(dateFrom)
    threshold_datetime_upper = pd.to_datetime(dateTo)
    df = df[df['date'] >= threshold_datetime_lower]
    df = df[df['date'] <= threshold_datetime_upper]

    #Pre-process
    preprocess = Text_preprocessing.Text_preprocessing(df)
    df = preprocess.preprocess()
    #Covert date to string
    df['date'] = df['date'].astype(str)
    return df


df = json_dataloader_toPandas(dataPath, dateFrom, dateTo)
tweets_ja= df['text'].tolist()
tweets_ja = tweets_ja[300:400]

#Japanese to english
#Translation model
# quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")# quantization_config=quantization_config)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "ja_XX"

start = time.process_time()
encoded_hi = tokenizer(tweets_ja, return_tensors="pt", truncation=True, padding= True)
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]).to(device)

translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(time.process_time() - start)
print(translated_text[0:9])
# 3. Update the DataFrame with the new value
# for index, value in df.iterrows():
#     df.at[index, 'text'] = translated_text[index]
#     print("Updating pandas")
#     if index > 8:
#         break

with open('fsd_1555_300_400.pkl', 'wb') as f:
    pickle.dump(translated_text, f)