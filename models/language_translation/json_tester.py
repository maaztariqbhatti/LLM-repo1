import json
import pickle
import pandas as pd
from openpyxl.workbook import Workbook

# x = [{"id": "t-1715518026806665548", "date": "2023-10-20T23:59:05Z", "text": "Three killed in Scotland as Storm Babet as major incidents declared and villages cut off by floods - The Independent Ireland \ud83c\uddee\ud83c\uddea  https://t.co/XahDpqZsfW", "tag_class": ["flood-affected"], "source": "twitter", "lang": "en", "urls": "https://twitter.com/portrigh/status/1715518026806665548", "locations": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [-4, 56]}, "properties": {"id": "g-2638360", "type": "adm4", "source": "geoparsing-enrichment", "parents": ["g-2638360", "g-2635167"], "name": "Scotland"}}]}, {"id": "t-1715517697323155773", "date": "2023-10-20T23:57:46Z", "text": "@BingoMotion @PaulEmbery @greateranglia Thanks for this information as I am not local (albeit not that far away) when I said that I cant see that all roads would be blocked due to flooding, I meant that you can surely go the long route around them &amp; come back into Norwich (eg via Cambridgeshire)", "tag_class": ["flood-logistics"], "source": "twitter", "lang": "en", "urls": "https://twitter.com/DiegoFuego71/status/1715517697323155773", "locations": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [0.08333, 52.33333]}, "properties": {"id": "g-2653940", "type": "adm5", "source": "geoparsing-enrichment", "parents": ["g-2653940", "g-2635167", "g-6269131"], "name": "Cambridgeshire"}}, {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1.29834, 52.62783]}, "properties": {"id": "g-2641181", "type": "adm7", "source": "geoparsing-enrichment", "parents": ["g-2641181", "g-2635167", "g-6269131", "g-2641455", "g-7290598"], "name": "Norwich"}}]}]

# with open("/home/mbhatti/mnt/d/LLM-repo1/models/language_translation/small_json.json", 'r') as json_file:
#     response_dict = json.load(json_file)

# texts = []
# dates = []

# for record in response_dict:
#     texts.append(record['text'])
#     dates.append(record['date'])

# # print(dates)

# for record in response_dict:
#     record['text'] = "Hello maaz here"
#     break
# print(response_dict)

# texts = []
# eng_text = []
# with open('fsd_1555_50.pkl', 'rb') as f:
#     texts.append(pickle.load(f))

# with open('fsd_1555_50_100.pkl', 'rb') as f:
#     texts.append(pickle.load(f))

# with open('fsd_1555_100_200.pkl', 'rb') as f:
#     texts.append(pickle.load(f))

# with open('fsd_1555_200_300.pkl', 'rb') as f:
#     texts.append(pickle.load(f))

# with open('fsd_1555_200_300.pkl', 'rb') as f:
#     texts.append(pickle.load(f))

# for i in texts:
#     for j in i :
#         eng_text.append(j)

# print(len(eng_text))
# print(eng_text[0:9])
#Join all the picke files together

#Download dataframe as csv
with open('/home/mbhatti/mnt/d/LLM-repo1/models/langchain_implementation/fsd_1555_0601_21:59:22_23:59:59.pkl', 'rb') as f:
   data = pickle.load(f)

df = pd.DataFrame(data)
df.to_excel("fsd_1555_0601_21:59:22_23:59:5.xlsx")
