import streamlit as st

# for getting random links
import scrapetube 
import random
#to convert numpy array to list
import numpy as np
## for getting title
import requests
from bs4 import BeautifulSoup

#for getting youtube length
from pytube import YouTube
#for bucketing
import embedding_bucketing.embedding_model_test as em
#own modules ao_core arch and config
from config import openai_api_key
import ao_core as ao
from arch_recommender import arch

# Initialize global variables
if "videos_in_list" not in st.session_state:
    st.session_state.videos_in_list = []
if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = []
if "current_binary_input" not in st.session_state:
    st.session_state.current_binary_input = []
if "training_history" not in st.session_state:
    st.session_state.training_history = (np.zeros([100,6], dtype="O"))
    st.session_state.numberVideos = 0
if "mood" not in st.session_state:
    st.session_state.mood = "Random"

display_video = False

#init agent
if "agent" not in st.session_state:
    print("-------creating agent-------")
    st.session_state.agent = ao.Agent(arch, notes=[])

    random_input_0 = np.random.randint(0, 2, st.session_state.agent.arch.Q__flat.shape, dtype=np.int8)
    random_input_1 = np.random.randint(0, 2, st.session_state.agent.arch.Q__flat.shape, dtype=np.int8)
    random_input_2 = np.random.randint(0, 2, st.session_state.agent.arch.Q__flat.shape, dtype=np.int8)
    random_input_3 = np.random.randint(0, 2, st.session_state.agent.arch.Q__flat.shape, dtype=np.int8)

    empty_label = np.zeros(st.session_state.agent.arch.Z__flat.shape, dtype=np.int8)
    full_label   = np.ones(st.session_state.agent.arch.Z__flat.shape, dtype=np.int8)

    st.session_state.agent.reset_state()
    st.session_state.agent.next_state(random_input_0, empty_label)
    st.session_state.agent.reset_state()
    st.session_state.agent.next_state(random_input_1, empty_label)
    st.session_state.agent.reset_state()
    st.session_state.agent.next_state(random_input_2,  full_label)
    st.session_state.agent.reset_state()
    st.session_state.agent.next_state(random_input_3,  full_label)
    st.session_state.agent.reset_state()


# Constants for embedding bucketing
max_distance = 20 # setting it high for no auto bucketing
amount_of_binary_digits = 10
type_of_distance_calc = "COSINE SIMILARITY"
start_Genre = ["Drama", "Comedy", "Action", "Romance", "Documentary", "Music", "Gaming", "Entertainment", "News", "Thriller", "Horror", "Science Fiction", "Fantasy", "Adventure", "Mystery", "Animation", "Family", "Historical", "Biography", "Superhero"
]
em.config(openai_api_key) # configuring openai client for embedding model
cache_file_name = "genre_embedding_cache.json"
cache, genre_buckets = em.init(cache_file_name, start_Genre)


st.set_page_config(page_title="DemoRS", layout="wide")

# Predefined list of random search terms
random_search_terms = ['funny', 'gaming', 'science', 'technology', 'news', 'random', 'adventure', "programming", "computer science"]


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
    soup = BeautifulSoup(r.text, "html.parser")

    link = soup.find_all(name="title")[0]
    title = str(link)
    title = title.replace("<title>","")
    title = title.replace("</title>","")
    print("title", title)
    return title

def get_FNF_from_title(title):
    input_message = ("Is this video title fiction or not"+ title)
    response = em.llm_call(input_message)
    response = response.upper() # Making the response upper case for no ambiguity
    fnf_binary = []
    if "FICTION" in response:
        fnf_binary = [1]
        response = "Fiction"
    else:
        fnf_binary = [0]
        response = "Non-fiction"
    return fnf_binary, response

def get_length_from_url(url): # returns if the video is short, medium or long in binary
    yt = YouTube(url)
    try:
        length = yt.length
    except Exception as e:
        print("error in getting length", e)
        length = 0
    length = round(length / 60, 2)
    length_binary = []
    if length < 5:
        length_binary = [0, 0]
    elif length >= 5 and length < 20:
        length_binary = [0, 1]
    else:
        length_binary = [1, 1]
    return length, length_binary

def get_video_data_from_url(url):
    length, length_binary = get_length_from_url(url)
    title = get_title_from_url(url)
    closest_genre, genre_binary_encoding = embedding_bucketing_response(title, max_distance, genre_buckets, type_of_distance_calc, amount_of_binary_digits)
    genre_binary_encoding = genre_binary_encoding.tolist()
    print("Closest genre to title", title, "is", closest_genre)
    fnf_binary, fnf = get_FNF_from_title(title)
    return length, length_binary, closest_genre, genre_binary_encoding, fnf, fnf_binary

def embedding_bucketing_response(uncategorized_input, max_distance, bucket_list, type_of_distance_calc, amount_of_binary_digits):
    sort_response = em.auto_sort(uncategorized_input, max_distance, bucket_list, type_of_distance_calc, amount_of_binary_digits) 

    closest_distance = sort_response[0]
    closest_bucket   = sort_response[1]  # which bucket the uncategorized_input was placed in
    bucket_id        = sort_response[2]  # the id of the closest_bucket
    bucket_binary    = sort_response[3]  # binary representation of the id for INPUT into api.aolabs.ai

    return closest_bucket, bucket_binary # returning the closest bucket and its binary encoding

def Get_mood_binary():
    mood = st.session_state.mood.upper()
    # converting mood to binary here
    if mood == "RANDOM":
        mood_binary = [1,0]
    if mood == "Serious":
        mood_binary = [1,1]
    if mood == "FUNNY":
        mood_binary = [0,1]
    else:
        mood_binary = [0,0] # if mood is not defined then give it 0,0
    return mood_binary, st.session_state.mood

def sort_agent_response(agent_response):
    #st.write("Agent response in binary: ", agent_response)
    count = 0
    for element in agent_response:
        if element == 1:  
            count += 1
    percentage = (count / len(agent_response)) * 100 
    return percentage
def next_video():  # function return closest genre and binary encoding of next video and displays it 
    display_video = False
    length, length_binary, closest_genre, genre_binary_encoding, fnf, fnf_binary = get_video_data_from_url(st.session_state.videos_in_list[0])
   
    mood_binary, mood = Get_mood_binary()
    st.write("Genre: ", closest_genre)
    st.write("Length: ", length)
    st.write("Fiction/Non-fiction: ", fnf)
    st.write("User's Mood: ", mood)
    st.write("")
    binary_input_to_agent = genre_binary_encoding+ length_binary + fnf_binary +mood_binary
   # st.write("binary input:", binary_input_to_agent)++
    st.session_state.current_binary_input = binary_input_to_agent # storing the current binary input to reduce redundant calls
    st.session_state.recommendation_result = agent_response(binary_input_to_agent)
    percentage_response = sort_agent_response(st.session_state.recommendation_result) 
    recommended = ("Chance recommended to you: "+ str(percentage_response) +"%")
    title = get_title_from_url(st.session_state.videos_in_list[0])
    temp_history = [title, recommended, closest_genre, length, fnf, mood]
    st.write("Recommendation result: ", recommended)
    st.session_state.training_history[st.session_state.numberVideos, :] = temp_history
    st.video(st.session_state.videos_in_list[0])
    st.session_state.numberVideos += 1
    return closest_genre, genre_binary_encoding

def train_agent(user_response):
    st.session_state.agent.reset_state()
    binary_input = st.session_state.current_binary_input
    if user_response == "pleasure":
        Cpos = True 
        Cneg = False
        label  = np.ones(st.session_state.agent.arch.Z__flat.shape, dtype=np.int8)
    elif user_response == "pain":
        Cneg = True
        Cpos = False
        label = np.zeros(st.session_state.agent.arch.Z__flat.shape, dtype=np.int8)
    # st.session_state.agent.next_state(INPUT=binary_input, Cpos=Cpos, Cneg=Cneg, print_result=False)
    st.session_state.agent.next_state(INPUT=binary_input, LABEL=label, print_result=False)


def agent_response(binary_input): # function to get agent response on next video
    #input = get_agent_input()
    st.session_state.agent.reset_state()
    st.session_state.agent.next_state( INPUT=binary_input, print_result=False)
    response = st.session_state.agent.story[st.session_state.agent.state-1, st.session_state.agent.arch.Z__flat]
    return response



# Title of the app
st.title("Recommender")

big_left, big_right = st.columns(2)

with big_left:
    st.session_state.mood = st.selectbox("Set your mood (as the user)", ("Random", "Funny", "Serious"))
    # Input for the number of links
    count = st.text_input("How many links to load", value='0')
    count = int(count) 
    url = st.text_input("Enter a youtube video to test", value=None)
    if url !=None:
        print("Adding url")
        try:
            st.session_state.videos_in_list.insert(0, url)
            next_video()
        except Exception as e:
            st.write("Error url not recognised")
    # Start button logic
    if st.button(f"Load {count} links"):
        if count > 0:
            
            for i in range(count):    
                data = get_random_youtube_link()
                while not data:  # Retry until a valid link is retrieved
                    data = get_random_youtube_link()
                if data not in st.session_state.videos_in_list:
                    st.session_state.videos_in_list.append(data)
            st.write(f"Loaded {count} videos.")
            display_video = True

with big_right:
    small_right, small_left = st.columns(2)
    with small_right:
        if st.button("Pleasure"):#
            train_agent(user_response="pleasure") # Train agent positively as user like recommendation
            if len(st.session_state.videos_in_list) > 0:
                st.session_state.videos_in_list.pop(0)  # Remove the first video from the list
                display_video = True
            else:
                st.write("The list is empty, cannot pop any more items.")

    with small_left:
        if st.button("Pain"):
            train_agent(user_response="pain") # train agent negatively as user dilike recommendation
            if len(st.session_state.videos_in_list) > 0:
                st.session_state.videos_in_list.pop(0)  # Remove the first video from the list
                display_video = True
            else:
                st.write("The list is empty, cannot pop any more items.")


    
    if display_video == True:
        genre, genre_binary_encoding = next_video()
    else:
        st.write("No more videos in the list.")

st.write("### Training History:")
st.write(st.session_state.training_history[0:st.session_state.numberVideos, :])