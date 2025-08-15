"""
Decision tree extraction from medical text.
"""
from typing import List, Tuple, Dict, TypedDict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class Message(TypedDict):
    role: str
    content: str


def construct_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer, GenerationConfig]:
    i = input("Warning. This will download model weights [roughly 15GB]. Proceed? (y/n)")
    if i.lower() != 'y':
        print("Model download aborted.")
        return None, None, None

    # The paper found Baichuan2-7B performed well. Other options include Llama-3, Mistral, etc.
    model_name = "baichuan-inc/Baichuan2-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="offload"
    )

    # Optional: You can define generation parameters
    generation_config = GenerationConfig(max_new_tokens=512)

    return model, tokenizer, generation_config

# Medical text from the article's example
tonic_clonic_seizures_example = """For patients with generalized tonic-clonic seizures, valproic acid is applicable. If not applicable, and the patient has myoclonic seizures or suspected juvenile myoclonic epilepsy, carbamazepine should not be used."""

def build_prompt(medical_text: str = tonic_clonic_seizures_example) -> List[Message]:
    # This prompt guides the model through the Chain-of-Thought process
    prompt_template = f"""
    You are an expert at extracting medical decision trees (MDTs) from text and formatting them.

    Here is the medical text:
    "{medical_text}"

    Follow these steps carefully:
    1.  First, identify all **Decision Points** (questions to be asked) and final **Outcomes** (conclusions or actions).
    2.  Second, structure this logic into a simple tree. Use "IF/ELSE IF/ELSE" for branches.

    EXAMPLE:
    Text: "For headaches, if the patient has a fever, check for meningitis. Otherwise, recommend rest."
    Output:
    DECISION POINT: Does the patient have a fever?
    - IF 'Yes': OUTCOME: Check for meningitis.
    - IF 'No': OUTCOME: Recommend rest.

    YOUR TASK:
    [Your thinking process starts here...]
    """

    # Structure the input as a list of messages
    messages: List[Message] = [
        Message(role="user", content=prompt_template)
    ]

    return messages


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
    import bs4

    soup = bs4.BeautifulSoup(html_content, 'html.parser')
    
    return soup.text

def fetch_html_via_url(url: str) -> str:
    """
    Fetch HTML content from a given URL.
    """
    import requests

    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch data from {url}, status code: {response.status_code}")


topic_map: Dict[str, str] = {
    'Atrial Fibrillation with Heart Failure': 'https://www.ahajournals.org/doi/10.1161/HAE.0000000000000078',
    'High Blood Pressure': 'https://www.ahajournals.org/doi/10.1161/hyp.0000000000000065'
}

if __name__ == "__main__":
    response = get_response()
    print(response)

    url = topic_map.get('High Blood Pressure')
    #html_content = fetch_html_via_url(url)
    #triplets = extract_from_html(html_content)
    #print(triplets)



# Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers
# pip install xformers.
