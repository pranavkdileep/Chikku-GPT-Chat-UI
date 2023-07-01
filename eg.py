import json
import glob
from gradio_client import Client
import os

def chat(question):
    client = Client("https://pranavkdileep-chikku-gpt-2-0.hf.space/")
    result = client.predict(
        question,  # Input text
        api_name="/bot"
    )

    # Find the latest JSON file in the directory
    json_files = glob.glob("/tmp/gradio/tmp*.json")
    latest_json_file = max(json_files, key=lambda x: os.path.getmtime(x))

    # Print the contents of the JSON file
    with open(latest_json_file) as file:
        json_data = json.load(file)
        answer = json_data[0][1]
        print(answer)
        return answer

chat("what is your name?")
