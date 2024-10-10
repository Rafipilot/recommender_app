import streamlit as st
import scrapetube
import random

## for getting title
import requests
from bs4 import BeautifulSoup

import embedding_bucketing.embedding_model_test as em
from config import openai_api_key

import ao_core as ao
from arch_recommender import arch

# Initialize global variables
if "videos_in_list" not in st.session_state:
    st.session_state.videos_in_list = []

display_video = False

if "agent" not in st.session_state:
    print("-------creating agent-------")
    st.session_state.agent = ao.Agent(arch, notes=[])

max_distance = 20 # setting it high for no auto bucketing
amount_of_binary_digits = 10
type_of_distance_calc = "COSINE SIMILARITY"
start_Genre = genres = [
    "Drama", 
    "Comedy", 
    "Action", 
    "Romance", 
    "Documentary", 
    "Music", 
    "Gaming", 
    "Entertainment", 
    "News", 
    "Thriller", 
    "Horror", 
    "Science Fiction", 
    "Fantasy", 
    "Adventure", 
    "Mystery", 
    "Animation", 
    "Family", 
    "Historical", 
    "Biography", 
    "Superhero"
]
em.config(openai_api_key) # configuring openai client for embedding model
cache_file_name = "genre_embedding_cache.json"
cache, genre_buckets = em.init(cache_file_name, start_Genre)


st.set_page_config(page_title="DemoRS", layout="wide")

# Predefined list of random search terms
random_search_terms = ['funny', 'gaming', 'science', 'technology', 'news', 'random', 'adventure', "programming", "computer science"]

def train_agent(input, user_response):  # function to train the agent
    if user_response == "pleasure":
        Cpos = True 
        Cneg = False
    elif user_response == "pain":
        Cneg = True
        Cpos = False
    
    st.session_state.agent.next_state(INPUT=input, Cpos=Cpos, Cneg=Cneg, print_result=False)


def get_random_youtube_link():
    print("Attempting to get a random video link...")
    
    # Select a random search term
    search_term = random.choice(random_search_terms)
    
    # Get videos from scrapetube
    videos = scrapetube.get_search(query=search_term, limit=10)
    
    # Shuffle and pick a random video
    video_list = list(videos)
    random.shuffle(video_list)
    
    if video_list:
        random_video = random.choice(video_list)
        video_id = random_video.get('videoId')
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    
    return None

def get_title_from_url(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text)

    link = soup.find_all(name="title")[0]
    title = str(link)
    title = title.replace("<title>","")
    title = title.replace("</title>","")
    print("title", title)
    return title

def embedding_bucketing_response(uncategorized_input, max_distance, bucket_list, type_of_distance_calc, amount_of_binary_digits):
    sort_response = em.auto_sort(uncategorized_input, max_distance, bucket_list, type_of_distance_calc, amount_of_binary_digits) 

    closest_distance = sort_response[0]
    closest_bucket   = sort_response[1]  # which bucket the uncategorized_input was placed in
    bucket_id        = sort_response[2]  # the id of the closest_bucket
    bucket_binary    = sort_response[3]  # binary representation of the id for INPUT into api.aolabs.ai

    return closest_bucket, bucket_binary # returning the closest bucket and its binary encoding


def next_video():
    display_video = False
    title = get_title_from_url(st.session_state.videos_in_list[0])
    closest_genre, genre_binary_encoding = embedding_bucketing_response(title, max_distance, genre_buckets, type_of_distance_calc, amount_of_binary_digits)
    print("Closest genre to title", title, "is", closest_genre)
    st.write("Genre:", closest_genre)
    st.video(st.session_state.videos_in_list[0])
    return closest_genre, genre_binary_encoding

# Title of the app
st.title("Recommender")

# Input for the number of links
count = st.text_input("How many links to load", value='0')
count = int(count) if count.isdigit() else 0  # Handle invalid input

# Start button logic
if st.button("Start"):
    if count > 0:
        st.write(f"Loading {count} links...")
        for i in range(count):
            data = get_random_youtube_link()
            while not data:  # Retry until a valid link is retrieved
                data = get_random_youtube_link()
            st.session_state.videos_in_list.append(data)
        st.write(f"Loaded {count} videos.")
        genre, genre_binary_encoding = next_video()
small_right, small_left = st.columns(2)

with small_right:
    if st.button("Pleasure"):
        if len(st.session_state.videos_in_list) > 0:
            st.session_state.videos_in_list.pop(0)  # Remove the first video from the list
            display_video = True
        else:
            st.write("The list is empty, cannot pop any more items.")

with small_left:
    if st.button("Pain"):
        if len(st.session_state.videos_in_list) > 0:
            st.session_state.videos_in_list.pop(0)  # Remove the first video from the list
            display_video = True
        else:
            st.write("The list is empty, cannot pop any more items.")


if st.session_state.videos_in_list:
    if display_video == True:
        genre, genre_binary_encoding = next_video()
else:
    st.write("No more videos in the list.")