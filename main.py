import streamlit as st

# for getting random links
import scrapetube 
import random
#to convert numpy array to list
import numpy as np
import pandas as pd
## for getting title
import requests
from bs4 import BeautifulSoup
import streamlit_analytics2

#for getting youtube length
from pytube import YouTube
import yt_dlp
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
    st.session_state.training_history = (np.zeros([100,7], dtype="O"))
    st.session_state.numberVideos = 0
if "mood" not in st.session_state:
    st.session_state.mood = "Random"
if "display_video" not in st.session_state:
    st.session_state.display_video = False

#init agent
if "agent" not in st.session_state:
    print("-------creating agent-------")
    st.session_state.agent = ao.Agent(arch, notes=[])

    random_input_0 = np.random.randint(0, 2, st.session_state.agent.arch.Q__flat.shape, dtype=np.int8)
    random_input_1 = np.random.randint(0, 2, st.session_state.agent.arch.Q__flat.shape, dtype=np.int8)
    random_input_2 = np.random.randint(0, 2, st.session_state.agent.arch.Q__flat.shape, dtype=np.int8)
    random_input_3 = np.random.randint(0, 2, st.session_state.agent.arch.Q__flat.shape, dtype=np.int8)

    random_label = np.random.randint(0,2, st.session_state.agent.arch.Z__flat.shape, dtype=np.int8)

    random_inputs = [random_input_0, random_input_1, random_input_2, random_input_3]

# intially train on random inputs 
    for i in range(4):
        st.session_state.agent.reset_state()
        st.session_state.agent.next_state(random_inputs[i], random_label)

# Constants for embedding bucketing
max_distance = 20 # setting it high for no auto bucketing
amount_of_binary_digits = 10
type_of_distance_calc = "COSINE SIMILARITY"
start_Genre = ["Drama", "Comedy", "Action", "Romance", "Documentary", "Music", "Gaming", "Entertainment", "News", "Thriller", "Horror", "Science Fiction", "Fantasy", "Adventure", "Mystery", "Animation", "Family", "Historical", "Biography", "Superhero"
]

em.config(openai_api_key) # configuring openai client for embedding model
print("configuring em")

if "cache" not in st.session_state:
    cache_file_name = "genre_embedding_cache.json"
    st.session_state.cache, st.session_state.genre_buckets = em.init(cache_file_name, start_Genre)
    print("init em")


# Predefined list of random search terms
random_search_terms = ['funny', 'gaming', 'science', 'technology', 'news', 'random', 'adventure', "programming", "computer science"]


def get_random_youtube_link():  
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
    # Initialize yt-dlp with options
    ydl_opts = {}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract video info
        video_info = ydl.extract_info(url, download=False)
        # Get the title from the video info
        title = video_info.get('title', None)
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
    closest_genre, genre_binary_encoding = embedding_bucketing_response(st.session_state.cache, title, max_distance, st.session_state.genre_buckets, type_of_distance_calc, amount_of_binary_digits)
    genre_binary_encoding = genre_binary_encoding.tolist()
    fnf_binary, fnf = get_FNF_from_title(title)
    return length, length_binary, closest_genre, genre_binary_encoding, fnf, fnf_binary

def embedding_bucketing_response(cache, uncategorized_input, max_distance, bucket_list, type_of_distance_calc, amount_of_binary_digits):
    sort_response = em.auto_sort(cache, uncategorized_input, max_distance, bucket_list, type_of_distance_calc, amount_of_binary_digits)  

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
    #display_video = False
    data = get_random_youtube_link()
    while not data:  # Retry until a valid link is retrieved
        data = get_random_youtube_link()
    if data not in st.session_state.videos_in_list:
        st.session_state.videos_in_list.append(data)
    st.session_state.display_video = True


    length, length_binary, closest_genre, genre_binary_encoding, fnf, fnf_binary = get_video_data_from_url(st.session_state.videos_in_list[0])
   
    mood_binary, mood = Get_mood_binary()
    st.markdown("     Genre: "+str(closest_genre), help="Extracted by an LLM")
    st.markdown("     Length: "+str(length), help="in minutes; extracted via pytube")
    st.markdown("     Fiction/Non-fiction: "+str(fnf), help="Extracted by an LLM")
    st.markdown("     User's Mood: "+str(mood),  help="Inputted by user")
    st.markdown("")
    binary_input_to_agent = genre_binary_encoding+ length_binary + fnf_binary +mood_binary
   # st.write("binary input:", binary_input_to_agent)++
    st.session_state.current_binary_input = binary_input_to_agent # storing the current binary input to reduce redundant calls
    st.session_state.recommendation_result = agent_response(binary_input_to_agent)
    percentage_response = sort_agent_response(st.session_state.recommendation_result) 
    recommended = (str(percentage_response) +"%")
    title = get_title_from_url(st.session_state.videos_in_list[0])
    temp_history = [title, closest_genre, length, fnf, mood, recommended, "User's Training"]
    print("temp history: ", temp_history)
    st.session_state.training_history[st.session_state.numberVideos, :] = temp_history
    st.write("**Agent's Recommendation:**  ", recommended)
    st.video(st.session_state.videos_in_list[0])
    return closest_genre, genre_binary_encoding

def train_agent(user_response):
    st.session_state.agent.reset_state()
    binary_input = st.session_state.current_binary_input
    if user_response == "RECOMMEND MORE":
        Cpos = True 
        Cneg = False
        label  = np.ones(st.session_state.agent.arch.Z__flat.shape, dtype=np.int8)
    elif user_response == "STOP RECOMMENDING":
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

streamlit_analytics2.start_tracking()
st.set_page_config(
    page_title="Recommender Demo by AO Labs",
    page_icon="misc/ao_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://discord.gg/Zg9bHPYss5",
        "Report a bug": "mailto:eng@aolabs.ai",
        "About": "AO Labs builds next-gen AI models that learn after training; learn more at docs.aolabs.ai/docs/mnist-benchmark",
    },
)

# Title of the app
st.title("LLM + Weightless NNs - a Continuously Learning Personal YouTube Recommender")
st.write("### *a preview by [aolabs.ai](https://www.aolabs.ai/)*")

big_left, big_right = st.columns([0.3, 0.7], gap="large")

with big_left:
    st.session_state.mood = st.selectbox("Set your mood (as the user):", ("Random", "Funny", "Serious"))
    st.divider()
    url = st.text_input("Enter link to a YouTube video: ", value=None, placeholder="Optional", help="This app automatically loads YouTube videos, and you can also add a specific YouTube link here.")
    # Input for the number of links
#    count = st.text_input("How many links to load", value='0')
#    count = int(count) 
    if url !=None:
        if st.button("Add Link"):
            try:
                if url not in st.session_state.videos_in_list:
                    st.session_state.videos_in_list.insert(0, url)
                    print(st.session_state.videos_in_list)
                else:
                    st.write("Unable to add link as it has already been used; please try another")
            except Exception as e:
                st.write("Error: URL not recognised; please try another")
            st.session_state.display_video = True

    
    # Start button logic (removed the button)

    data = get_random_youtube_link()
    while not data:  # Retry until a valid link is retrieved
        data = get_random_youtube_link()
    if data not in st.session_state.videos_in_list:
        st.session_state.videos_in_list.append(data)

    st.divider()
    with st.expander("### Agent's Training History"):
        history_titles = ["Title", "Closest Genre", "Duration", "Type", "User's Mood", "Agent's Recommendation", "User's Training" ]
        df = pd.DataFrame(
            st.session_state.training_history[0:st.session_state.numberVideos, :], columns=history_titles
            )
        st.dataframe(df)

with big_right:
    small_right, small_left = st.columns(2)
    if small_right.button(":green[RECOMMEND MORE]", type="primary", icon=":material/thumb_up:"):#
        train_agent(user_response="RECOMMEND MORE") # Train agent positively as user like recommendation
        if len(st.session_state.videos_in_list) > 1:
            st.session_state.videos_in_list.pop(0)  # Remove the first video from the list
            st.session_state.display_video = True
            st.session_state.training_history[st.session_state.numberVideos, -1] = "User Liked"
        # else:
        #     st.write("The list is empty, cannot pop any more items.")
        #     st.session_state.display_video = False  #Stop displaying the video once we have run out of videos
        st.session_state.numberVideos += 1
    if small_left.button(":red[STOP RECOMMENDING]", icon=":material/thumb_down:"):
        train_agent(user_response="STOP RECOMMENDING") # train agent negatively as user dislike recommendation
        if len(st.session_state.videos_in_list) > 1:
            st.session_state.videos_in_list.pop(0)  # Remove the first video from the list
            st.session_state.display_video = True
            st.session_state.training_history[st.session_state.numberVideos, -1] = "User Disliked"
        # else:
        #     st.write("The list is empty, cannot pop any more items.")
        #     st.session_state.display_video = False  #Stop displaying the video once we have run out of videos
        st.session_state.numberVideos += 1
        
    # if st.session_state.display_video == True:
    genre, genre_binary_encoding = next_video()

st.write("---")
footer_md = """
    [View & fork the code behind this application here.](https://github.com/aolabsai/Recommender) \n
    To learn more about Weightless Neural Networks and the new generation of AI we're developing at AO Labs, [visit our docs.aolabs.ai.](https://docs.aolabs.ai/)\n
    \n
    We eagerly welcome contributors and hackers at all levels! [Say hi on our discord.](https://discord.gg/Zg9bHPYss5)
    """
st.markdown(footer_md)
st.image("misc/aolabs-logo-horizontal-full-color-white-text.png", width=300)

streamlit_analytics2.stop_tracking()