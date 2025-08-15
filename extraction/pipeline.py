"""
Pipeline helper functions.
"""
from typing import List, Tuple

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


def extract_from_html(html_content: str) -> List[Tuple[str, str, str]]:
    """
    Extract text from HTML content.
    """
    soup = bs4.BeautifulSoup(html_content, 'html.parser')
    
    return soup.text


def fetch_html_via_url(url: str) -> str:
    """
    Fetch HTML content from a given URL.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch data from {url}, status code: {response.status_code}")
