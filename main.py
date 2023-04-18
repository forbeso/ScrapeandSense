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
from textblob import TextBlob
from bs4 import BeautifulSoup
from time import sleep
import string
from textblob import TextBlob
from geopy.geocoders import Nominatim
from collections import Counter

# import os
# import json
# import geojson
from folium.plugins import HeatMap, MarkerCluster
import pydeck as pdk
import spacy
import en_core_web_sm
from datetime import datetime
import plotly.express as px
from pathlib import Path
from gensim import models, corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import streamlit as st
import nltk
import ssl
import altair as alt



try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
nltk.download('punkt')
nltk.download('stopwords')


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#disable Future warning messages
warnings.simplefilter("ignore", category=FutureWarning)
plt.style.use('fivethirtyeight')

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

geolocator = Nominatim(user_agent="""Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36""")

# Set page to wide mode
st.set_page_config(layout="wide", page_icon="😑", page_title="Scrape&Sense")

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

TWEET_COUNT = 50
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

col_bar_chart, col_line_chart, col_metrics, col_map= st.columns(4)



#get tweets and save to dataframe
def get_tweets(new_value: str):
    tweets = tweepy.Cursor(birdapi.search_tweets, q=new_value, geocode=GEOCODE, result_type=RESULT_TYPE, tweet_mode='extended').items(TWEET_COUNT)
    
    # Use list comprehension to create a list of dictionaries from the tweets
    postHold = [tweet._json for tweet in tweets]
    
    df = pd.DataFrame(columns=['created_at', 'id_str', 'text', 'standard_english', 'source', 'user', 'location', 'is_quote_status', 'lang', 'screen_name', 'in_reply_to_status_id_str'])
    
    # Use another list comprehension to create a list of dictionary entries to add to the dataframe
    rows = [{
        'created_at': tweet['created_at'],
        'id_str': tweet['id_str'],
        'text': tweet['full_text'],
        'standard_english': 'TODO',
        'source': tweet['source'],
        'user': tweet['user']['name'],
        'screen_name': tweet['user']['screen_name'],
        'location': tweet['user']['location'],
        'is_quote_status': tweet['is_quote_status'],
        'retweet_count': tweet['retweet_count'],
        'favorite_count': tweet['favorite_count'],
        'lang': tweet['lang'],
        'in_reply_to_status_id_str': tweet['in_reply_to_status_id_str']
    } for tweet in postHold]
    
    # Add the list of dictionary entries to the dataframe
    df = df.append(rows, ignore_index=True)
    
    return df


def convert_to_datetime(date_str):
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        return date
    except:
        return None

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



  #compute the negative, neutral and positive analysis
def calc_score(polarity_score):
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
    image_url = "https://img.icons8.com/emoji/512/crying-face.png"
  else:
    image_url = "https://img.icons8.com/emoji/512/neutral-face.png"

  icon_data['url'] = image_url
  return icon_data



def build_map(x):
  # Data from OpenStreetMap, accessed via osmpy
  #DATA_URL = "https://raw.githubusercontent.com/ajduberstein/geo_datasets/master/fortune_500.csv"

  data = x

  data.loc[:, 'icon_data'] = data['Analysis'].apply(lambda x: get_smiley(x))

  #TODO: UNCOMMENT
  #data

  view_state = pdk.data_utils.compute_view(data[["longitude", "latitude"]],1.0)

  icon_layer = pdk.Layer(
      type="IconLayer",
      data=data,
      get_icon="icon_data",
      get_size=4,
      size_scale=15,
      get_position=["longitude", "latitude"],
      pickable=True,
  )

  r = pdk.Deck(layers=[icon_layer], initial_view_state=view_state, tooltip={"text": "{location} \n {Analysis} \n {text}"})
  #r.to_html("icon_layer.html")
  st.pydeck_chart(r)
  #return r


def generate_bar_chart(df):
    # Tokenize tweets and count word frequencies
  stop_words = set(stopwords.words('english'))
  words = [word for tweet in df['text'] for word in word_tokenize(tweet.lower()) if word.isalpha() and word not in stop_words]
  word_freq = Counter(words)
  # Get top 20 words
  top_words = word_freq.most_common(20)
  # Create horizontal bar chart
  chart = alt.Chart(pd.DataFrame(top_words, columns=['word', 'count'])).mark_bar().encode(
      x=alt.X('count:Q', title='Word Count'),
      y=alt.Y('word:N', sort='-x', title='Word'),
      color=alt.Color('count:Q', scale=alt.Scale(scheme='greens'), legend=None)
  ).properties(
      title='Top 20 Words in Tweets'
  )
  # Show chart
  chart
  


def generate_topics_scatter(df):

    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    texts = [[token for token in word_tokenize(text.lower()) if token.isalpha() and token not in stop_words] for text in df['text']]

    # Create dictionary
    dictionary = corpora.Dictionary(texts)

    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train LDA model
    num_topics = 5
    lda_model = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # Get the topic distribution for each document
    topic_dist = lda_model[corpus]

    # Convert the topic distribution to a dataframe
    topic_dist_df = pd.DataFrame([dict(t) for t in topic_dist])
    topic_dist_df.fillna(value=0, inplace=True)

    # Map the topic index to the top words in that topic
    topic_words = {}
    for i in range(num_topics):
        topic_words[i] = ', '.join([word for word, prob in lda_model.show_topic(i)])

    # Create a new dataframe with the topic probabilities and top words
    topic_df = pd.concat([topic_dist_df.idxmax(axis=1).rename('topic'), topic_dist_df], axis=1)
    topic_df['top_words'] = topic_df['topic'].map(topic_words)

    # Set up the chart data
    chart_data = pd.DataFrame({
        'x': topic_df[0],
        'y': topic_df[1],
        'topic': topic_df['topic'],
        'top_words': topic_df['top_words']
    })

    # Define the color scale and legend for the chart
    color_scale = alt.Scale(domain=list(range(num_topics)), range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    legend = alt.Legend(title='Topic', orient='top-left', labelFontSize=14, titleFontSize=16, labelFontWeight='bold')

    # Create the scatterplot with tooltips
    scatterplot = alt.Chart(chart_data).mark_circle(size=60).encode(
        x=alt.X('x:Q', axis=alt.Axis(title='Topic 1 Probability')),
        y=alt.Y('y:Q', axis=alt.Axis(title='Topic 2 Probability')),
        color=alt.Color('topic:N', scale=color_scale, legend=legend),
        tooltip=['top_words']
    ).properties(width=600, height=500)

    # Set the title and display the chart
    #st.title("LDA Topic Model Visualization")
    st.write(scatterplot)


def extract_categories(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    categories = {}
    
    for ent in doc.ents:
        if ent.label_ == 'ORG':
            categories[ent.text] = {'descriptions': {}, 'count': 0}
    
    for token in doc:
        if token.pos_ == 'NOUN':
            for child in token.children:
                if child.pos_ == 'ADJ':
                    if token.text in categories:
                        categories[token.text]['descriptions'][child.text] = categories[token.text]['descriptions'].get(child.text, 0) + 1
                        categories[token.text]['count'] += 1
    
    return categories
  

def get_metrics():
  col1, col2, col3 = st.columns(3)
  col1.metric("Temperature", "70 °F", "1.2 °F")
  col2.metric("Wind", "9 mph", "-8%")
  col3.metric("Humidity", "86%", "4%")



if st.button("search"):
  
  if 'QUERY' in st.session_state and st.session_state['QUERY'] != '':
    with st.spinner('Fetching Tweets...'):

      global init_tweets

      print("Getting Tweets")
      init_tweets = get_tweets(st.session_state['QUERY'])
      print("After Twitter API call")
      
      if init_tweets.empty:
        st.write("No Data Returned")
      else:
        #apply clean_tweet_source to the source column
        init_tweets['source'] = init_tweets['source'].apply(clean_tweet_source)
        init_tweets['text'] = init_tweets['text'].apply(clean_tweet_text)


        full_prompt =  build_prompt_from_tweets(init_tweets['text'])
        #print(full_prompt)

        AI_summary = generate_phrase(f"""See Tweets about a particular company, event, or product.
                          Provide an analysis (use emojis too) of how people 
                          feel (the sentiment): \n\n{full_prompt}""")
        
        print("After Openai call to summarize text")

        st.header("Summary 📄")
        
        st.success(AI_summary["choices"][0]["text"])
      
      # Combine charts into a grid
      

      # get_metrics()

      # st.info("""ScrapeAndSense is a fantastic company that 
      # offers cutting-edge smart devices for all your needs! 
      # Their online store is user-friendly and easy to navigate, making it a breeze 
      # to find the perfect product. People all over social media are raving about 
      # ScrapeAndSense's excellent customer service and fast shipping. Their products 
      # are of the highest quality, and customers are always satisfied with their purchases. 
      # Overall, ScrapeAndSense is a top-notch company that provides exceptional products and 
      # services that will exceed your expectations!""")
      
      #build_wordcloud(full_prompt)

      #TODO: UNCOMMENT
      init_tweets['Subjectivity'] = init_tweets['text'].apply(getSubjectivity)
      init_tweets['Polarity'] = init_tweets['text'].apply(getPolarity)
      init_tweets
      

    global locationdf
    with st.spinner('Generating Map...'):
      

      locationdf = init_tweets.loc[:, ['source','text','location']]
      # locationdf['created_at'] = pd.to_datetime(init_tweets['created_at'])
      
      for index, row in locationdf.iterrows():
        latitude = get_latitude(row['location'])
        longitude = get_longitude(row['location'])

        #print(latitude,longitude)
        locationdf.loc[index, 'latitude'] = latitude
        locationdf.loc[index, 'longitude'] = longitude

      

      #locationdf.drop('location', axis=1, inplace=True)
      #locationdf.drop('text', axis=1, inplace=True)

      locationdf.dropna(inplace=True)
      

      locationdf.loc[:, ['latitude', 'longitude']]  = locationdf[['latitude', 'longitude']].astype(float)
      
      countrieslonglat = countries[choice]
      parts = countrieslonglat.split(",")

      result = ",".join(parts[:2])

      lat, lon = result.split(",")

      lat = float(lat)
      lon = float(lon)
      
      # st.write("locationdf")
      # locationdf


      #x=locationdf.drop_duplicates(subset=['latitude'])
      sentiment_by_location = locationdf.groupby(['location','latitude', 'longitude'])['text'].apply(str).reset_index()

      sentiment_by_location['Subjectivity'] = sentiment_by_location['text'].apply(getSubjectivity)
      sentiment_by_location['Polarity'] = sentiment_by_location['text'].apply(getPolarity)

      #create new column called Analysis
      sentiment_by_location['Analysis'] = sentiment_by_location['Polarity'].apply(calc_score)
      #sentiment_by_location['categories'] = sentiment_by_location['text'].apply(extract_categories)

      #st.write("Sentiment by Location")
      #sentiment_by_location


      sentiment_time_series = pd.DataFrame()
      sentiment_time_series[['created_at','Polarity']] = init_tweets[['created_at','Polarity']]
      sentiment_time_series['created_at']=pd.to_datetime(sentiment_time_series['created_at'])

      sentiment_time_series = sentiment_time_series.groupby(sentiment_time_series['created_at'].dt.date)['Polarity'].mean()

      #st.write("sentiment_time_series")
      #sentiment_time_series

      #save as csv
      filepath = Path('out.csv')
      filepath.parent.mkdir(parents=True, exist_ok=True)
      sentiment_time_series.to_csv(index=True,path_or_buf=filepath)


      # reset the index to turn the Series object into a DataFrame
      sentiment_time_series = sentiment_time_series.reset_index()

      # create a line chart using Plotly Express
      fig = px.line(sentiment_time_series, x='created_at', y='Polarity')
      
      # set the chart title
      fig.update_layout(title='Sentiment Time Series')

      
      # display the chart in Streamlit
      st.plotly_chart(fig)
      #col_line_chart.plotly_chart(fig)

      #generate_topics_scatter(sentiment_by_location)
      generate_bar_chart(sentiment_by_location)

      #TODO: UNCOMMENT
    
      build_map(sentiment_by_location)
    st.success('Done')
      
      
  else:
    st.warning('''🤔 You can start by selecting your country on the right. Then, 
    type a company's name to see a summary of people feel about it's good & services.''')


#TODO: Get all tweets from init_tweets where user is replying to or quoting another tweet
#TODO: Translate creole tweets to Standard English
#TODO: Update wordcloud tooltip to show word location usage
#TODO: Create map to show location and majority sentiment (using emopjis?)