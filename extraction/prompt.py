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
    Text: "For a patient presenting with acute chest pain, the initial assessment must prioritize life-threatening conditions. If the pain is crushing and radiates to the arm or jaw, suspect Acute Myocardial Infarction (AMI) and begin MONA protocol; however, if the patient has a known aspirin allergy, use clopidogrel instead. If the pain is sharp, pleuritic, and accompanied by shortness of breath, a Pulmonary Embolism (PE) is a primary concern, and a CT angiogram should be ordered. If the pain is described as burning, is worse when lying down, and is related to meals, it suggests Gastroesophageal Reflux Disease (GERD), for which a trial of antacids is the first step. For all other presentations of chest pain, a standard workup with an EKG and cardiac enzymes is warranted."
    Output:
______
DECISION POINT: What are the characteristics of the patient's acute chest pain?
    IF 'Crushing and radiating to the arm or jaw':
        DECISION POINT: Does the patient have a known aspirin allergy?
            IF 'Yes':
                OUTCOME: Use clopidogrel and begin MONA protocol for suspected AMI.
            IF 'No':
                OUTCOME: Use aspirin and begin MONA protocol for suspected AMI.
    IF 'Sharp, pleuritic, and with shortness of breath':
        OUTCOME: Order a CT angiogram to investigate for Pulmonary Embolism.
    IF 'Burning, worse when lying down, and related to meals':
        OUTCOME: Recommend a trial of antacids for suspected GERD.
    IF 'Other':
        OUTCOME: Perform a standard workup with an EKG and cardiac enzymes.
______

    YOUR TASK:
    [Your thinking process starts here...]
    """

    # Structure the input as a list of messages
    messages: List[Message] = [
        Message(role="user", content=prompt_template)
    ]

    return messages
