# Recommender System
### Author & Maintainer: Rafipilot
Maintainer: [Rafipilot](https://github.com/Rafipilot), rafayel.latif@gmail.com


## Description
This is a basic video recommender system designed to offer personalized video recommendations. Unlike many modern systems that rely on collaborative filtering—resulting in suggestions based on broad user trends—this system aims to provide more unique and individually tailored recommendations. By giving users greater control over how they provide feedback, the system helps break free from repetitive content and exposes users to a wider range of videos, making it easier to find content that truly resonates with their tastes.

## Installation/Setup

### Local Environment
If you plan to run the app in a conda or virtual environment, make sure to set up your environment following the respective instructions for those tools.

1. Install the requirements:

    ```bash
    pip install -r requirements.txt
    ```

2. Install ao_core and ao_arch with the pip install git+ method which lets you install python code from git repos.

    ```bash
    pip install git+https://github.com/aolabsai/ao_arch git+https://github.com/aolabsai/ao_core
    ```

3. Run the application with the following command:

    ```bash
    streamlit run recommender.py
    ```

4. Once running, the app will be accessible at `localhost:8501`.


### Docker Installation

1) Generate a GitHub Personal Access Token to ao_core    
    Go to https://github.com/settings/tokens?type=beta

2) Clone this repo and create a `.env` file in your local clone where you'll add the PAT as follows:
    `ao_github_PAT=token_goes_here`
    No spaces! See `.env_example`.

3) In a Git Bash terminal, build and run the Dockerfile with these commands:
```shell
export DOCKER_BUILDKIT=1

docker build --secret id=env,src=.env -t "ao_app" .

docker run -p 8501:8501 streamlit
```
You're done! Access the app at `localhost:8501` in your browser.

## Usage

The recommender system works by loading a set of random video links. Once the user hits the Run button, a video will be shown, and the system will suggest whether it recommends the video or not. The user can then provide feedback using "pain" or "pleasure" signals to guide the recommendation process. Based on this feedback, the system adjusts its responses and suggests another video. This cycle continues, allowing for more accurate and personalized recommendations over time.


## Contributing

Fork the repository, make your changes, and submit a pull request for review. 



