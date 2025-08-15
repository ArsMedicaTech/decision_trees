"""
This class is responsible for parsing the output from the LLM and converting it into a structured format.
"""
from typing import Any, Dict, List, Tuple

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
        self.decision_pattern = re.compile(r"^(?P<indent>\s*)DECISION POINT:\s*(?P<question>.*)")
        self.if_pattern = re.compile(r"^(?P<indent>\s*)IF\s*'(?P<condition>[^']+)'\s*:")
        self.outcome_pattern = re.compile(r"^(?P<indent>\s*)OUTCOME:\s*(?P<outcome>.*)")
    
    def get_indent(self, line: str) -> int:
        """Returns the indentation level of a line."""
        return len(line) - len(line.lstrip(' '))

    def parse(self, llm_output: str) -> Dict[Any, Any]:
        """
        Parses the full multi-line output from the LLM.
        """
        llm_output = sanitize_string(llm_output).expandtabs(4)
        lines = llm_output.strip().split('\n')
        
        if not lines:
            return {}
        
        # The main call to the recursive parser starts at index 0.
        tree, _ = self._parse_node(lines, 0)
        return tree
    
    def _parse_node(self, lines: List[str], start_idx: int) -> Tuple[Dict[str, Any], int]:
        """
        Recursively parses a node and its children, respecting indentation.
        Returns the parsed node and the index of the next line to process.
        """
        # --- 1. Get the base indentation and question for this node ---
        base_indent = self.get_indent(lines[start_idx])
        question_line = lines[start_idx].strip()
        question = question_line.replace('DECISION POINT:', '').strip()
        
        node = {"question": question, "branches": {}}
        
        # --- 2. Iterate through subsequent lines to find children ---
        current_idx = start_idx + 1
        while current_idx < len(lines):
            line = lines[current_idx]
            line_indent = self.get_indent(line)
            
            # --- 3. If a line is not indented further, this node is finished ---
            if line_indent <= base_indent and line.strip():
                break # Return control to the parent call
            
            if_match = self.if_pattern.match(line)
            if if_match:
                condition = if_match.group("condition")
                
                # Look ahead to the next line to see what kind of child it is
                next_line_idx = current_idx + 1
                if next_line_idx < len(lines):
                    next_line = lines[next_line_idx]
                    
                    # Child is an OUTCOME
                    outcome_match = self.outcome_pattern.match(next_line)
                    if outcome_match:
                        node["branches"][condition] = outcome_match.group("outcome")
                        current_idx += 2 # Move past the IF and the OUTCOME
                        continue
                    
                    # Child is a nested DECISION POINT
                    decision_match = self.decision_pattern.match(next_line)
                    if decision_match:
                        # Recursively parse the nested node
                        nested_node, end_idx = self._parse_node(lines, next_line_idx)
                        node["branches"][condition] = nested_node
                        current_idx = end_idx # Jump past the entire parsed nested node
                        continue

            # If the line is not a child or is unparsable, just move to the next one
            current_idx += 1

        # --- 4. Return the completed node and the next index to process ---
        return node, current_idx
    
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
