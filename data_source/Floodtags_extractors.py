import floodtags_API
import datetime

class Floodtags_extractor:
    """
    Provide json flood tags reponse. Function extracts useful information only
    """
    def json_evidenceParser(self, evidence_dict : dict) -> dict:

        assert isinstance(evidence_dict, dict), \
                'Expected input "evidence_dict" to be a dict, but a %s was input.'%type(evidence_dict)

        relevant_text_dicts = []
        if not evidence_dict:
            return relevant_text_dicts

        for tag in evidence_dict["data"]["tags"]:

            relevant_text_dicts.append({'id':tag['id'],
                                'date': tag['date'],
                                'text': tag['text'],
                                'tag_class': tag['classes'] if 'classes' in tag else None,
                                'source': tag['source']['type'],
                                'lang': tag['source']['lang'],
                                # 'urls': tag['source']['url'],
                                'locations': tag['locations']})

        return relevant_text_dicts
    
    """
    Utilise flood tags API to extract information based on the paramters provided and save response to PDF
    """
    def csv_evidenceParser(self):
        start_date  = datetime(2024, 1, 2)
        end_date = datetime(2024, 1, 4)
        sources = ["english-flood",
                    "english-flood-news"]

        flood_classes = ["class:flood-logistics",
                            "class:flood-affected"]

        #Bounding box of the area of interest
        #UK
        bbox = ('United Kingdom', (-7.57216793459, 49.959999905, 1.68153079591, 58.6350001085))

        watershed_gdf = floodtags_API.analyze_watersheds(start_date=start_date,
                                            end_date=end_date,
                                            sources= sources,
                                            bbox= bbox[1])

        #Save file to csv
        interval = bbox[0] + '_' + start_date.strftime("%Y-%m-%d") + '_' + end_date.strftime("%Y-%m-%d")
        watershed_gdf.to_csv('csvImports/{0}.csv'.format(interval), header=True)
        

