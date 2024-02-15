# -*- coding: utf-8 -*-
import time
import os
from typing import List, Optional, Tuple, Union
from functools import cache
import requests
from datetime import datetime
import dotenv
from dataclasses import dataclass, field
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry.base import BaseGeometry
# from shapely.geometry import Point, LineString, MultiPoint, Polygon, MultiPolygon
import warnings

@dataclass
class FloodTagConfig:

    api_url : str = "https://api.floodtags.com/v2/"

    tags_url : str = api_url + "tags"

    tag_export_end_point : str = tags_url + "/export" #GET
    tag_edit_end_point : str = tags_url + "/edit" #POST
    tag_delete_end_point : str = tags_url + "/delete" #POST

    events_url : str = api_url + "events"

    event_export_end_point : str = events_url + "/export" #GET
    event_edit_end_point: str = events_url + "/edit" #POST
    event_delete_end_point: str = events_url + "/delete" #POST

    users_url : str = api_url + "users"

    user_export_end_point : str = users_url + "/export" #GET
    user_edit_end_point: str = users_url + "/edit" #POST
    user_delete_end_point: str = users_url + "/delete" #POST

    request_timeout_limit_seconds : int = 600

    query_intervals : List[float] =  field(default_factory = lambda : [0.1, 1.0, 10.0, 60.0, 300.0, 300.0, 300.0]) #seconds

    def __post_init__(self):

        for url in  (self.tags_url, self.events_url, self.users_url):
            assert isinstance(url, str), 'Expected all input urls to be strings, but a %s was input.'%type(url)

        for end_point in (self.tag_export_end_point,
                         self.tag_edit_end_point,
                         self.tag_delete_end_point,
                         self.event_edit_end_point,
                         self.event_export_end_point,
                         self.event_delete_end_point,
                         self.user_edit_end_point,
                         self.user_delete_end_point,
                         self.user_export_end_point):
            assert isinstance(end_point, str), 'Expected all endpoints to be strings but a %s was input.'%type(end_point)

        #assert isinstance(self.sources, list), 'Expected input "sources" to be a list, but a %s was input.'%type(self.source)
        #assert isinstance(self.flood_classes, list), 'Expected input "flood_classes" to be a list, but a %s was input.'%type(self.flood_classes)

        #assert len(self.sources) > 0, 'Expected input "sources" to be a list of strings, but an empty list was input.'
        #assert len(self.flood_classes) > 0, 'Expected input "flood_classes" to be a list of strings, but an empty list was input.'

        # for item in self.sources:
        #     assert isinstance(item, str), 'Expected input "sources" to be a list of strings, but a %s element was found.'%type(item)
        # for item in self.flood_classes:
        #     assert isinstance(item, str), 'Expected input "flood_classes" to be a list of strings, but a %s element was found.'%type(item)

        assert isinstance(self.request_timeout_limit_seconds, int), 'Expected input "request_timeout_limit_seconds" to be an integer, but a %s was input.'%type(self.request_timeout_limit_seconds)
        assert self.request_timeout_limit_seconds > 0, 'Expected input "request_timeout_limit_seconds" to be a positive integer, but a %s was input.'%self.request_timeout_limit_seconds

        assert isinstance(self.query_intervals, list), 'Expected input "query_intervals" to be a list but a %s was input.'%type(self.query_intervals)
        assert len(self.query_intervals) > 0, 'Expected input "query_intervals" to have elements.'
        for item in self.query_intervals:
            assert isinstance(item, (int, float)), 'Expected each element in input "query_intervals" to be either an integer or a float but a %s was found'%type(item)
            assert item >= 0.0, 'Expected each element in input "query_intervals" to be greater or equal to zero.'

        # dotenv.load_dotenv()
        # if flood_tag_api_key := os.getenv("FLOOD_TAG_API_KEY"):
        #     self.flood_tag_api_key : str = flood_tag_api_key
        # else:
        #     raise Exception("Please specify the FLOOD_TAG_API_KEY env variable")
        self.flood_tag_api_key : str = "7b6f7181-fc27-4b64-a533-0c390927788d"
class FloodTagsRetriever:

    def __init__(self, config: FloodTagConfig = FloodTagConfig()):
        assert isinstance(config, FloodTagConfig), (
                'Expected input "config" to be a FloodTagConfig object but a %s was input.'%type(config))
        self.config : FloodTagConfig = config
        self.stored_output = None

    def get_tag_information(self,
                            tag_ids: Optional[Union[pd.Series, List[str]]] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            aoi_geom : Optional[BaseGeometry] = None,
                            sources : Optional[List[str]] = None,
                            flood_classes : Optional[List[str]] = None,
                            tags_paginator : int = 0) -> Optional[dict]:

        def _get_payload(tag_ids: Optional[Union[pd.Series, List[str]]],
                         start_date: Optional[datetime],
                         end_date: Optional[datetime],
                         aoi_geom : Optional[BaseGeometry],
                         sources : Optional[List[str]],
                         flood_classes : Optional[List[str]],
                         tags_paginator : int) -> dict:

            payload = {"views":"tags",
                        "view.tags.limit":100,
                       "view.tags.skip": tags_paginator,
            }

            if tag_ids is None:
                assert aoi_geom is not None, 'If "tag_ids" is None, "aoi_geom" must be provided.'
                assert start_date is not None, 'If "tag_ids" is None, "start_date" must be provided.'
                assert end_date is not None, 'If "tag_ids" is None, "end_date" must be provided.'
                assert sources is not None, 'If "tag_ids" is None, "sources" must be provided.'
                for item in sources:
                    assert isinstance(item, str), 'Expected input "sources" to be a list of strings, but a %s element was found.'%type(item)
                payload['sources'] = ",".join(sources)
            else:
                if isinstance(tag_ids, (pd.Series, gpd.GeoSeries)):
                    tag_ids = tag_ids.tolist()
                    payload['ids'] = ','.join(tag_ids)
                elif isinstance(tag_ids, str):
                    payload['ids'] = tag_ids
                elif not isinstance(tag_ids, list):
                    raise TypeError('Expected input "tag_ids" to be either None or a list of strings but a %s was input.' % type(tag_ids))

                num_tags = len(tag_ids)
                if num_tags == 0:
                    return None
                if num_tags > 50:
                    raise ValueError('The FloodTags API can only handle max 50 tags at a time.')

                for item in tag_ids:
                    assert isinstance(item, str), (
                            'Expected input "tag_ids" to be a list or series of strings but a %s element was found.' % type(item))

            if start_date is not None:
                assert isinstance(start_date, datetime), \
                    'Expected input "start_date" to be a datetime object but a %s was input.' % type(start_date)
                payload['date.since'] = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            if end_date is not None:
                assert isinstance(end_date, datetime), \
                    'Expected input "end_date" to be a datetime object but a %s was input.' % type(end_date)
                payload['date.until'] = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

            if aoi_geom is not None:
                assert isinstance(aoi_geom, list)
                payload['location.bbox'] = ",".join([str(coord) for coord in aoi_geom])
            
            if flood_classes is not None:
                for item in flood_classes:
                    assert isinstance(item, str), 'Expected input "flood_classes" to be a list of strings, but a %s element was found.'%type(item)
                payload['filters'] = ",".join(flood_classes)

            return payload

        payload = _get_payload(tag_ids=tag_ids,
                               start_date=start_date,
                               end_date=end_date,
                               aoi_geom=aoi_geom,
                               sources=sources,
                               flood_classes= flood_classes,
                               tags_paginator=tags_paginator)

        output = None
        for next_timeout in self.config.query_intervals:
            try:
                req_response = requests.get(self.config.tags_url,
                                            params=payload,
                                            headers={"Authorization": f'FloodTags-Key key="{self.config.flood_tag_api_key}"'},
                                            timeout=self.config.request_timeout_limit_seconds,
                                            )
                req_response.raise_for_status()
            except requests.exceptions.HTTPError:
                warnings.warn(
                    'Connection error. Code %d - Message : %s. Waiting for %.1f seconds and attempting again.' % (
                    req_response.status_code,
                    str(req_response.content),
                    next_timeout
                    ))
                time.sleep(next_timeout)
            except (requests.exceptions.ConnectTimeout,
                    requests.exceptions.ReadTimeout):
                warnings.warn('Connection timed out. Waiting for %.1f seconds and attempting again.' % next_timeout)
                time.sleep(next_timeout)
            else:
                output = req_response.json()
                break

        if output is None:
            print("No data was retrieved from FloodTags.")
            warnings.warn('Connection error. No more attempts will be made. Returning empty results.')
        else:
            if self.stored_output == None:
                self.stored_output = output
            else:
                self.stored_output['data']['tags'].extend(output['data']['tags'])

            if output['total'] > (tags_paginator+100):#payload['view.tags.limit']:

                #To ensure API call doesnt get jammed
                time.sleep(2)

                new_output = self.get_tag_information(tag_ids=tag_ids,
                                                    start_date=start_date,
                                                    end_date=end_date,
                                                    sources=sources,
                                                    aoi_geom=aoi_geom,
                                                    tags_paginator=tags_paginator + payload['view.tags.limit'])

                # output['data']['tags'].extend(new_output['data']['tags'])
                # output['total'] += new_output['total']
                #self.stored_output.extend(new_output['data']['tags'])

        # full_output = self.stored_output
        return self.stored_output

def analyze_watersheds(start_date : datetime,
                       end_date : datetime,
                       bbox : Tuple,
                       sources : List[str],
                       flood_classes : List[str] = None) -> pd.DataFrame:

    flood_tags_retriever = FloodTagsRetriever()
    
    # aoi_geometry = shapely.geometry.box(*bbox, ccw=True)
    tag_information_json = flood_tags_retriever.get_tag_information(start_date=start_date,
                                                                    end_date=end_date,
                                                                    aoi_geom=bbox,
                                                                    sources =sources,
                                                                    flood_classes = flood_classes)

    tag_information_pd = _parse_text_evidence(tag_information_json)

    return tag_information_pd

def _parse_text_evidence(evidence_dict : dict) -> dict:
    '''Find any relevant text evidence strings'''

    assert isinstance(evidence_dict, dict), \
            'Expected input "evidence_dict" to be a dict, but a %s was input.'%type(evidence_dict)

    text_dicts = []
    if not evidence_dict:
        return text_dicts

    for tag in evidence_dict["data"]["tags"]:

        text_dicts.append({'id':tag['id'],
                            'date': tag['date'],
                            'text': tag['text'],
                            'tag_class': tag['classes'] if 'classes' in tag else None,
                            'source': tag['source']['type'],
                            'lang': tag['source']['lang'],
                            'urls': tag['source']['url'],
                            'locations': tag['locations']})

    # df = pd.DataFrame(text_dicts)
    # print(df.head())
    # print(df.size)
    return text_dicts

if __name__ == "__main__":

    # event_id = 736

    start_date  = datetime(2023, 10, 16)
    end_date = datetime(2023, 10, 18)
    sources = ["english-flood"]
    

    # flood_classes = ["class:flood-logistics",
    #                  "class:flood-affected"]
    
    
    #Bounding box of the area of interest
    bbox = [-5.855844,51.49784,2.369591,57.742844]

    response_dict = analyze_watersheds(start_date=start_date,
                                       end_date=end_date,
                                       sources= sources,
                                       bbox= bbox)

    