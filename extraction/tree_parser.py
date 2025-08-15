"""
This class is responsible for parsing the output from the LLM and converting it into a structured format.
"""
from typing import Any, Dict, List

import re
import json


def sanitize_string(text: str) -> str:
    """
    Programmatically replaces non-breaking spaces (\xa0) with regular spaces
    to protect against copy-paste errors.
    """
    return text.replace('\xa0', ' ')


class LLMTreeParser:
    """
    Parses the semi-structured text output from an LLM into a nested dictionary
    that conforms to the arsmedicatech decision tree schema.

    NOTE: One might ask why we are not directing the original extractor LLM to
    simply return structured outputs, but the LLM is much more specialized and
    this allows us to relieve some of the burden on the extraction process.
    """
    
    def __init__(self):
        pass
    
    def parse(self, llm_output: str) -> Dict[Any, Any]:
        """
        Parses the full multi-line output from the LLM.

        Returns:
            A dictionary representing the structured decision tree.
        """
        llm_output = sanitize_string(llm_output).expandtabs(4)
        lines = llm_output.strip().split('\n')
        
        # Parse the decision tree structure
        tree = self._parse_tree(lines)
        return tree
    
    def _parse_tree(self, lines: List[str]) -> Dict[str, Any]:
        """
        Parse the entire decision tree using indentation levels.
        """
        if not lines:
            return {}
        
        # Find the root decision point
        root_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('DECISION POINT:'):
                root_idx = i
                break
        
        if root_idx is None:
            return {}
        
        return self._parse_node(lines, root_idx)
    
    def _parse_node(self, lines: List[str], start_idx: int) -> Dict[str, Any]:
        """
        Parse a decision node starting at start_idx.
        """
        # Extract the question
        question_line = lines[start_idx].strip()
        question = question_line.replace('DECISION POINT:', '').strip()
        
        # Find all IF blocks that belong to this decision node
        branches = {}
        current_idx = start_idx + 1
        
        while current_idx < len(lines):
            line = lines[current_idx].strip()
            
            # Skip empty lines
            if not line:
                current_idx += 1
                continue
            
            # Check if this is an IF line
            if line.startswith('IF '):
                # Extract the condition (e.g., 'Yes', 'No')
                match = re.search(r"IF\s*'([^']+)'\s*:", line)
                if match:
                    condition = match.group(1)
                    
                    # Look ahead to see what comes after this IF
                    next_idx = current_idx + 1
                    if next_idx < len(lines):
                        next_line = lines[next_idx].strip()
                        
                        if next_line.startswith('OUTCOME:'):
                            # This is a simple outcome
                            outcome_text = next_line.replace('OUTCOME:', '').strip()
                            branches[condition] = outcome_text
                            current_idx = next_idx + 1
                        elif next_line.startswith('DECISION POINT:'):
                            # This is a nested decision node
                            nested_tree = self._parse_node(lines, next_idx)
                            branches[condition] = nested_tree
                            # Skip to after the nested tree
                            current_idx = self._find_end_of_node(lines, next_idx)
                        else:
                            # Skip this IF block if we can't parse it
                            current_idx += 1
                    else:
                        current_idx += 1
                else:
                    current_idx += 1
            elif line.startswith('DECISION POINT:'):
                # We've hit another decision point at the same level, so we're done with this one
                break
            else:
                current_idx += 1
        
        return {
            "question": question,
            "branches": branches
        }
    
    def _find_end_of_node(self, lines: List[str], start_idx: int) -> int:
        """
        Find the end of a decision node starting at start_idx.
        This looks for the next decision point at the same indentation level.
        """
        current_idx = start_idx + 1
        while current_idx < len(lines):
            line = lines[current_idx].strip()
            if line.startswith('DECISION POINT:'):
                # Found another decision point, so we're done
                break
            current_idx += 1
        return current_idx


# --- Example Usage ---
def test():
    # The semi-structured text we expect from the LLM
    llm_output_text_simple = """
DECISION POINT: Is valproic acid applicable?
    IF 'Yes':
        OUTCOME: Prescribe valproic acid.
    IF 'No':
        OUTCOME: Refer to neurologist.
"""
    
    print("Simple test:")
    print(repr(llm_output_text_simple))

    llm_output_text = """
DECISION POINT: Does the patient have generalized tonic-clonic seizures?
    IF 'Yes':
        DECISION POINT: Is valproic acid applicable?
        IF 'Yes':
            OUTCOME: Prescribe valproic acid.
        IF 'No':
            OUTCOME: Refer to neurologist.
    IF 'No':
        DECISION POINT: Does the patient have myoclonic seizures?
        IF 'Yes':
            OUTCOME: Do not use carbamazepine.
        IF 'No':
            OUTCOME: Further evaluation needed.
"""

    # Test with a more complex nested structure
    llm_output_text_complex = """
DECISION POINT: What type of seizure does the patient have?
    IF 'Generalized':
        DECISION POINT: Is it tonic-clonic?
        IF 'Yes':
            DECISION POINT: Is valproic acid contraindicated?
            IF 'Yes':
                OUTCOME: Use alternative medication.
            IF 'No':
                OUTCOME: Prescribe valproic acid.
        IF 'No':
            OUTCOME: Consider other generalized seizure medications.
    IF 'Focal':
        DECISION POINT: Is it simple or complex?
        IF 'Simple':
            OUTCOME: Monitor and consider carbamazepine.
        IF 'Complex':
            OUTCOME: Refer to specialist.
    IF 'Unknown':
        OUTCOME: Conduct EEG and neurological evaluation.
"""

    # One more complex nested example
    llm_output_text_complex_2 = """
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
"""

    print("\nComplex test:")
    print(repr(llm_output_text))

    print("\nVery complex test:")
    print(repr(llm_output_text_complex))

    print("\nVery complex test:")
    print(repr(llm_output_text_complex_2))

    # Create a parser and process the text
    parser = LLMTreeParser()
    
    print("\n--- Simple Tree ---")
    decision_tree_simple = parser.parse(llm_output_text_simple)
    print(json.dumps(decision_tree_simple, indent=4))
    
    print("\n--- Complex Tree ---")
    decision_tree_complex = parser.parse(llm_output_text)
    print(json.dumps(decision_tree_complex, indent=4))
    
    print("\n--- Very Complex Tree ---")
    decision_tree_very_complex = parser.parse(llm_output_text_complex)
    print(json.dumps(decision_tree_very_complex, indent=4))

    print("\n--- Very Complex Tree 2 ---")
    decision_tree_very_complex_2 = parser.parse(llm_output_text_complex_2)
    print(json.dumps(decision_tree_very_complex_2, indent=4))


if __name__ == "__main__":
    test()


def parser_util(llm_output: str) -> Dict[Any, Any]:
    """
    Utility function to parse LLM output into a structured format.
    """
    parser = LLMTreeParser()
    return parser.parse(llm_output)
