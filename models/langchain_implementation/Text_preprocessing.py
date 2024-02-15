import re
import json
import emoji
import pandas as pd

class Text_preprocessing:
    def __init__(self, _df : pd) -> None:
        self.df = _df
    
    def emoji_remove(self, _text):
        """Remove emojis from text"""
        _text= emoji.demojize(_text)
        _text= re.sub(r'(:[!_\-\w]+:)', '', _text)
        return _text
    
    def preprocess(self) -> pd:
        for index,row in self.df.iterrows():
            text = row['text']

            """Remove URLs"""
            text =  re.sub(r"http\S+", "", text)

            """Remove user mentions"""
            text =  re.sub(r'@\w+', "", text)

            """Remove emoji"""
            text = self.emoji_remove(text)

            """Remove line breaks"""
            text = text.replace("\n", " ")

            """Remove multiple spaces with a single space"""
            text = re.sub(' +', " ", text)

            """Remove leading and trailing space from the tweets"""
            text = text.strip()

            #Update data frame
            self.df.loc[index, 'text'] = text

        return self.df

if __name__ == '__main__':

    # Load json and extract relevant records in pandas df
    with open("G:/My Drive/LLM-repo1/Floodtags_analytics/FSD1777_Oct23.json", 'r') as json_file:
        response_dict = json.load(json_file)

    # Convert to pandas df    
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(response_dict)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(columns=['id','tag_class', 'source', 'lang', 'urls','locations'])

    #Get data between thresholds
    threshold_datetime_lower = pd.to_datetime('2023-10-19 23:55:41+00:00')
    threshold_datetime_upper = pd.to_datetime('2023-10-19 23:58:47+00:00')
    df = df[df['date'] >= threshold_datetime_lower]
    df = df[df['date'] <= threshold_datetime_upper]

    #Pre-process
    preprocess = Text_preprocessing(df)
    df = preprocess.preprocess()

    for index,row in df.iterrows():
        print(row['text'])
    



#     text_raw = """Its almost midnight here in the uk, 3 minutes to be exact.

# The torrential rain and heavy winds have been non-stop all day, but really kicked up a notch at 6pm, and is still going strong now.

# If you are going about in the uk tonight, please drive safely, watch our for flooding."""

#     preprocess = Text_preprocessing(text_raw)
#     print(preprocess.preprocess())

    
