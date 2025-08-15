"""
Prompt builder.
"""
from typing import List, TypedDict


class Message(TypedDict):
    role: str
    content: str


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
