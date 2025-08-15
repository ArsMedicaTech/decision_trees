"""
This class is responsible for parsing the output from the LLM and converting it into a structured format.
"""
from typing import Any, Dict

import re
import json

class LLMTreeParser:
    """
    Parses the semi-structured text output from an LLM into a nested dictionary
    that conforms to the arsmedicatech decision tree schema.

    NOTE: One might ask why we are not directing the original extractor LLM to
    simply return structured outputs, but the LLM is much more specialized and
    this allows us to relieve some of the burden on the extraction process.
    """
    def __init__(self):
        # Compile regex patterns for efficiency
        self.decision_pattern = re.compile(r"^\s*DECISION POINT: (.*)")
        self.if_pattern = re.compile(r"^\s*- IF '(.*)':")
        self.outcome_pattern = re.compile(r"^\s*OUTCOME: (.*)")
        
        self.root = {}
        # The stack will hold references to the 'branches' dictionary at each level
        self.stack = []

    def get_indent_level(self, line: str) -> int:
        """Calculates the indentation level based on leading spaces."""
        return len(line) - len(line.lstrip(' '))

    def parse(self, llm_output: str) -> Dict[Any, Any]:
        """
        Parses the full multi-line output from the LLM.
        
        Returns:
            A dictionary representing the structured decision tree.
        """
        lines = llm_output.strip().split('\n')
        
        # Initialize with the root of the tree
        self.stack.append(self.root)
        current_branch_key = None
        
        for line in lines:
            if not line.strip():
                continue

            # Match against our patterns
            match_decision = self.decision_pattern.match(line)
            match_if = self.if_pattern.match(line)
            match_outcome = self.outcome_pattern.match(line)

            # Get the current parent's branches from the top of the stack
            parent_branches = self.stack[-1]

            if match_decision:
                question_text = match_decision.group(1).strip()
                new_node = {
                    "question": question_text,
                    "branches": {}
                }
                # If this is the very first node, it becomes the root
                if not parent_branches:
                    self.root.update(new_node)
                # Otherwise, it's a child of the previous 'IF'
                else:
                    parent_branches[current_branch_key] = new_node
                
                # We are now inside the new node's branches, so push them to the stack
                self.stack.append(new_node["branches"])

            elif match_if:
                # This line defines the key for the next branch
                current_branch_key = match_if.group(1).strip()
                # Pop from the stack to go back to the parent level before starting a new branch
                if len(self.stack) > 1: # Don't pop the root
                    self.stack.pop()
                
            elif match_outcome:
                outcome_text = match_outcome.group(1).strip()
                # The outcome is a leaf node in the current parent's branches
                parent_branches[current_branch_key] = outcome_text

        return self.root

# --- Example Usage ---
def test():
    # The semi-structured text we expect from the LLM
    llm_output_text = """
    DECISION POINT: Does the patient have generalized tonic-clonic seizures?
    - IF 'Yes':
        DECISION POINT: Is valproic acid applicable?
        - IF 'Yes':
            OUTCOME: Prescribe valproic acid.
        - IF 'No':
            OUTCOME: Refer to neurologist.
    - IF 'No':
        DECISION POINT: Does the patient have myoclonic seizures?
        - IF 'Yes':
            OUTCOME: Do not use carbamazepine.
        - IF 'No':
            OUTCOME: Further evaluation needed.
    """

    # Create a parser and process the text
    parser = LLMTreeParser()
    decision_tree = parser.parse(llm_output_text)

    # Print the resulting structured dictionary
    print(json.dumps(decision_tree, indent=4))


def parser_util(llm_output: str) -> Dict[Any, Any]:
    """
    Utility function to parse LLM output into a structured format.
    """
    parser = LLMTreeParser()
    return parser.parse(llm_output)
