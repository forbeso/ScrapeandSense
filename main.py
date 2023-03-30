#!/usr/bin/env python
# coding: utf-8

import tweepy
import streamlit_wordcloud as wordcloud
import pandas as pd
import numpy as np
import re
import openai
import matplotlib.pyplot as plt
import warnings
import streamlit as st
import sys
from textblob import TextBlob
from bs4 import BeautifulSoup
from stop_words import get_stop_words
from time import sleep
import string
import folium
from langdetect import detect
from textblob import TextBlob
from geopy.geocoders import Nominatim
from folium import Map, Choropleth
from streamlit_folium import folium_static
import pathlib as pl
import os
import json
import geojson
from folium.plugins import HeatMap, MarkerCluster
import pydeck as pdk
import random

#disable Future warning messages
warnings.simplefilter("ignore", category=FutureWarning)
plt.style.use('fivethirtyeight')


geolocator = Nominatim(user_agent="""Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36""")


#set up access

CONSUMER_KEY = 'fIDHPjQg0QaCEMDjCKEpURIEM'
CONSUMER_SECRET = 'Kn2y7W6i16vJUha70ObMzx3aqqJ3ur8hAekS576oGg9majM0HI'
ACCESS_KEY = '1045452666120351744-UofuQ8nmkMMYg6UyvIfS8vh7VeQclq'
ACCESS_SECRET = '1R0pMSUmnvR4ANgFB9d2RquYDLDbmhJQURBSUxNkEpba4'

openai.api_key = "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"

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
birdapi = tweepy.API(auth)

if 'QUERY' not in st.session_state:
  st.session_state['QUERY'] = ''

if 'COUNTRY' not in st.session_state:
  st.session_state['COUNTRY'] = ''

countries = {
  "Jamaica": ("18.19892,-77.32021,58.64mi"),
  "New York": ("40.80559,-73.25877,42.72mi"),

}
#print(countries['Jamaica'])



TWEET_MODE = 'extended'

TWEET_COUNT = 40
RESULT_TYPE = 'recent'
INCLUDE_ENTITIES = False

options = ['Jamaica', 'New York']
st.sidebar.title("ScrapeAndSense")
choice = st.sidebar.selectbox(label="Select Country", options=options)
#st.title("Creolex")

if choice in countries:
    GEOCODE = countries[choice]
    st.session_state['COUNTRY'] = choice
else:
    st.write(f"Invalid choice: {choice}")
#print(st.session_state['COUNTRY'])



#streamlit

svp = st.text_input(
  #key=st.session_state['QUERY'],
  label="Service Provider",
  value='',
  max_chars=25,
  help="Enter the name of the company you would like to view sentiments for.")


st.session_state['QUERY'] = svp.capitalize()

st.write(f"Company entered: {st.session_state['QUERY']}")


#get tweets and save to dataframe
def get_tweets(new_value: str):
  #global init_tweets 

  tweets = tweepy.Cursor(birdapi.search_tweets,q=new_value,geocode=GEOCODE,result_type=RESULT_TYPE,tweet_mode='extended').items(TWEET_COUNT)
  
  #loop and and each list to the array postHold
  postHold = []
  for tweet in tweets:
    postHold.append(tweet._json)  #get _json object from response
  #print(postHold)

  # create an empty dataframe to store the parsed data
  df = pd.DataFrame(columns=[
    'created_at', 'id_str', 'text', 'standard_english','source', 'user', 'location',
    'is_quote_status', 'lang', 'screen_name', 'in_reply_to_status_id_str'
  ])

  pd.set_option('display.max_colwidth', None)
  pd.set_option('display.max_rows', None)

  # loop over each tweet in the json data and add it to the dataframe
  for tweet in postHold:
    # extract user name and screen name
    user_name = tweet['user']['name']
    screen_name = tweet['user']['screen_name']
    location = tweet['user']['location']
    # add a new row to the dataframe
    df = df.append(
      {
        'created_at': tweet['created_at'],
        'id_str': tweet['id_str'],
        'text': tweet['full_text'],
        'standard_english': 'TODO',
        'source': tweet['source'],
        'user': user_name,
        'screen_name': screen_name,
        'location': location,
        'is_quote_status': tweet['is_quote_status'],
        'retweet_count': tweet['retweet_count'],
        'favorite_count': tweet['favorite_count'],
        'lang': tweet['lang'],
        'in_reply_to_status_id_str': tweet['in_reply_to_status_id_str']
      },
      ignore_index=True)
  #st.write(df)

  return df


def clean_tweet_source(a_tag: str):
  soup = BeautifulSoup(a_tag, 'html.parser')
  
  return soup.a.text.strip()


def clean_tweet_text(text: str):

  text = re.sub(r'https?:\/\/\S+', '', text)  #Remove the hyperlink
  text = re.sub(r'@[A-Za-z0-9]+', '',text)  #removing mentions - #the r tells python that this is a raw string
  text = re.sub(r'#', '', text)  #removing hashtags
  text = re.sub(r'RT[\s]+', '', text)  #removing the RT (retweets)

  return text


# def clean_location(text:str):

#   if text != '':
#     text = text.capitalize()

  
def build_prompt_from_tweets(column):
  prompt = ""

  try:
    for i in column:
      prompt = prompt + "\n" + i + "\n"
    #print(prompt)
    return prompt
  except Exception as err:
    st.error(f"Something went wrong: {err}")


def generate_phrase(prompt: str):
  response = openai.Completion.create(engine="text-davinci-003",
                                      prompt=prompt,
                                      max_tokens=2000,
                                      temperature=0.8)
  #sleep(5)
  #print(response)
  #st.write(response["choices"][0]["text"])
  return response


def word_count(text):
  # Remove punctuation and convert to lowercase
  cleaned_text = text.translate(str.maketrans('', '', string.punctuation))

  # Split the cleaned text into a list of words
  words_list = cleaned_text.split()

  # Initialize an empty dictionary
  word_count_dict = {}

  # Loop through the words and update the dictionary
  for word in words_list:
    if word in word_count_dict:
      word_count_dict[word] += 1
    else:
      word_count_dict[word] = 1

  # Return the word count dictionary
  return word_count_dict


#create wordcloud

def build_wordcloud(text):

  stop_words = [
    'a', 'an', 'and', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in',
    'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'with', 'i','am','my',
  ]

  #stop_words = get_stop_words('en')
  #print(stop_words)

  # Create a regular expression pattern to match the stop words
  pattern = r'\b(?:{})\b'.format('|'.join(stop_words))

  # Remove the stop words using re.sub()
  text = re.sub(pattern, '', text, flags=re.IGNORECASE)

  word_count_dict = word_count(text)

  wordcloud_bank = []
  for word in word_count_dict:
    wordcloud_bank.append(
      dict(text=word,
           value=word_count_dict[word],
           color="#b5de2b",
           count=word_count_dict[word],
           location=''))
  

  final_wordcloud = wordcloud.visualize(wordcloud_bank,
    tooltip_data_fields={
      'text': 'Word',
      'count': '# of Mentions',
      'value': 'Location mostly used'
      
    },
    per_word_coloring=False,
    palette="viridis",
    layout="archimedean")
  #print(wordcloud_bank)
  return final_wordcloud


def get_latitude(location):
    try:
        address = geolocator.geocode(location, timeout=None)
        return address.latitude
    except:

        return None

def get_longitude(location):
    try:
        address = geolocator.geocode(location, timeout=None)
        return address.longitude
    except:
        return None


#TODO: Create function to calculate sentiment score based on whether tweets are positive, neutral, or negative
#positive:1, neutral:0, negative:-1

#Create a function to get the subjectivity of text
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

#Create a fucntion to get the polarity of text
def getPolarity(text):
  return TextBlob(text).sentiment.polarity


def calc_score(polarity_score):
   #compute the negative, neutral and positive analysis
  if (polarity_score < 0):
    return 'Negative'
  elif polarity_score == 0:
    return 'Neutral'
  else:
    return 'Positive'


def get_smiley(sentiment):
  icon_data = {
        
        "width": 242,
        "height": 242,
        "anchorY": 242,
    }

  if sentiment == 'Positive':
    image_url = "https://img.icons8.com/emoji/512/slightly-smiling-face.png"
  elif sentiment=='Negative':
    image_url = "https://img.icons8.com/emoji/512/face-with-symbols-on-mouth.png"
  else:
    image_url = "https://img.icons8.com/emoji/512/confused-face.png"

  icon_data['url'] = image_url
  return icon_data



def build_map(x, lat,lon):
  # Data from OpenStreetMap, accessed via osmpy
  #DATA_URL = "https://raw.githubusercontent.com/ajduberstein/geo_datasets/master/fortune_500.csv"

  data = x

  data.loc[:, 'icon_data'] = data['Analysis'].apply(lambda x: get_smiley(x))

  #TODO: UNCOMMENT
  #data

  view_state = pdk.data_utils.compute_view(data[["longitude", "latitude"]],0.5)

  icon_layer = pdk.Layer(
      type="IconLayer",
      data=data,
      get_icon="icon_data",
      get_size=4,
      size_scale=15,
      get_position=["longitude", "latitude"],
      pickable=True,
  )

  r = pdk.Deck(layers=[icon_layer], initial_view_state=view_state, tooltip={"text": "{location}"})
  #r.to_html("icon_layer.html")
  st.pydeck_chart(r)


  

if st.button("search"):
  with st.spinner('Fetching tweets and generating map'):
    if 'QUERY' in st.session_state and st.session_state['QUERY'] != '':

      global init_tweets
      init_tweets = get_tweets(st.session_state['QUERY'])
      print("After Twitter API call")
      
      if init_tweets.empty:
        st.write("No Data Returned")
      else:
        #apply clean_tweet_source to the source column
        init_tweets['source'] = init_tweets['source'].apply(clean_tweet_source)
        init_tweets['text'] = init_tweets['text'].apply(clean_tweet_text)

        init_tweets['Subjectivity'] = init_tweets['text'].apply(getSubjectivity)
        init_tweets['Polarity'] = init_tweets['text'].apply(getPolarity)

        #create new column called Analysis
        init_tweets['Analysis'] = init_tweets['Polarity'].apply(calc_score)
        

        full_prompt =  build_prompt_from_tweets(init_tweets['text'])
        #print(full_prompt)

        AI_summary = generate_phrase(f"""See Tweets about a particular company's services.
                          Provide an analysis (use emojis too) of how people 
                          feel (the sentiment) about the companies services: \n\n{full_prompt}""")
        
        print("After Openai call to summarize text")

        st.success(AI_summary["choices"][0]["text"])
        #show_map()
        #build_wordcloud(full_prompt)

        #TODO: UNCOMMENT
        #init_tweets

        if not init_tweets.empty:
          st.write(init_tweets.describe())

        global locationdf
        locationdf = init_tweets.loc[:, ['text','location','Analysis']]
        
        for index, row in locationdf.iterrows():
          latitude = get_latitude(row['location'])
          longitude = get_longitude(row['location'])

          #print(latitude,longitude)
          locationdf.loc[index, 'latitude'] = latitude
          locationdf.loc[index, 'longitude'] = longitude

        

        #locationdf.drop('location', axis=1, inplace=True)
        locationdf.drop('text', axis=1, inplace=True)

        locationdf.dropna(inplace=True)
        

        locationdf.loc[:, ['latitude', 'longitude']]  = locationdf[['latitude', 'longitude']].astype(float)

        countrieslonglat = countries[choice]
        parts = countrieslonglat.split(",")

        result = ",".join(parts[:2])

        lat, lon = result.split(",")

        lat = float(lat)
        lon = float(lon)
        
        #locationdf
        x=locationdf.drop_duplicates(subset=['latitude'])
        # show_map(lat=lat, lon=lon)
        # Example usage
      

        #TODO: UNCOMMENT
        #x

        build_map(x, lat, lon)
      
        
        
    else:
      st.warning('''🤔 You can start by selecting your country on the right. Then, 
      type a company's name to see a summary of people feel about it's good & services.''')



# def show_map():

#   df = pd.DataFrame.from_dict(locationdf, orient='index', columns=['coordinates'])

#   df[['latitude', 'longitude', 'distance']] = df['coordinates'].str.split(',', expand=True)
  
#   df.drop('coordinates', axis=1, inplace=True)
#   df.drop('distance', axis=1, inplace=True)

#   df[['latitude', 'longitude']] = df[['latitude', 'longitude']].astype(float)

#   st.write("Map showing...")
#   st.map(df)


#TODO: Get all tweets from init_tweets where user is replying to or quoting another tweet
#TODO: Translate creole tweets to Standard English
#TODO: Update wordcloud tooltip to show word location usage
#TODO: Create map to show location and majority sentiment (using emopjis?)