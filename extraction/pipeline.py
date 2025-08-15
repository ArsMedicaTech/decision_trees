"""
Pipeline helper functions.
"""
from typing import Dict, List, Tuple

import json

from extraction.model import construct_model
from extraction.prompt import build_prompt

import bs4
import requests


def get_response() -> str:
    model, tokenizer, generation_config = construct_model()
    if model is None:
        return "Model construction failed."

    messages = build_prompt()

    # Call the built-in chat method
    response_text = model.chat(tokenizer, messages, generation_config=generation_config)

    return response_text


def extract_from_html(html_content: str) -> str:
    """
    Extracts structured text from HTML, preserving paragraph breaks.
    """
    soup = bs4.BeautifulSoup(html_content, 'html.parser')
    
    # Find all meaningful text blocks (paragraphs, headers, list items)
    text_blocks = [tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])]
    
    # Join them with double newlines to maintain structure for chunking
    return "\n\n".join(text_blocks)


def fetch_html_via_url(url: str) -> str:
    """
    Fetch HTML content from a given URL.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch data from {url}, status code: {response.status_code}")


def load_topic_map() -> Dict[str, str]:
    """
    Load the topic map from a JSON file.
    """
    with open("extraction/topic_map.json", "r") as f:
        topic_map = json.load(f)
    return topic_map
