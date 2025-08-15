"""
This class is responsible for parsing the output from the LLM and converting it into a structured format.
"""
from typing import Any, Dict

import re
import json

from lark import Lark, Transformer, v_args


def sanitize_string(text: str) -> str:
    """
    Programmatically replaces non-breaking spaces (\xa0) with regular spaces
    to protect against copy-paste errors.
    """
    return text.replace('\xa0', ' ')


decision_tree_grammar = r"""
    ?tree: decision_node

    ?decision_node: DECISION_POINT NEWLINE (_INDENT if_block+ _DEDENT) -> build_decision_node

    ?if_block: IF_LINE NEWLINE (_INDENT (outcome | decision_node) _DEDENT) -> build_if_block

    ?outcome: OUTCOME NEWLINE -> get_outcome_text

    // --- Terminal Tokens ---
    DECISION_POINT: "DECISION POINT:" /[^\n]+/
    IF_LINE: /IF\s*'[^']+'\s*:/
    OUTCOME: "OUTCOME:" /[^\n]+/

    // --- Whitespace and Newlines ---
    %import common.WS
    %import common.NEWLINE
    %ignore WS

    // --- Indentation Handling ---
    %declare _INDENT _DEDENT
"""

@v_args(inline=True)
class TreeToJson(Transformer):
    def build_decision_node(self, question, *branches):
        # This method is called for a 'decision_node' rule.
        # It receives the question text and all the processed child branches.
        return {
            "question": question.strip(),
            "branches": dict(branches) # Convert list of (key, value) tuples into a dict
        }

    def build_if_block(self, if_line, child_node):
        # This method is called for an 'if_block' rule.
        # It receives the condition text (like 'Yes') and the processed child (either an outcome or another decision node).

        # Extract the quoted key (e.g., 'Yes') from the IF_LINE token
        match = re.search(r"'(.*?)'", if_line)
        key = match.group(1) if match else None
        return key, child_node

    def get_outcome_text(self, text):
        # This just cleans up the outcome text.
        return text.strip()

    # --- Token Processing ---
    # These methods clean up the raw token values before they are passed to the rule methods above.
    def DECISION_POINT(self, s):
        return s.value.strip()
    
    def QUOTED_STRING(self, s):
        return s[1:-1]

    def IF_LINE(self, s):
        # We pass the whole token string to the transformer method
        return s.value

    def OUTCOME(self, s):
        return s.value.strip()


class LLMTreeParser:
    """
    Parses the semi-structured text output from an LLM into a nested dictionary
    that conforms to the arsmedicatech decision tree schema.

    NOTE: One might ask why we are not directing the original extractor LLM to
    simply return structured outputs, but the LLM is much more specialized and
    this allows us to relieve some of the burden on the extraction process.
    """
    def __init__(self):
        # Create the parser instance with our grammar.
        # The 'start' rule is 'tree', and we tell it to use Lark's indentation lexer.
        clean_grammar = sanitize_string(decision_tree_grammar)

        self.parser = Lark(
            clean_grammar,
            start='tree',
            parser='lalr',
            lexer='contextual'
        )
        
        self.transformer = TreeToJson()
    
    def parse(self, llm_output: str) -> Dict[Any, Any]:
        """
        Parses the full multi-line output from the LLM.

        Returns:
            A dictionary representing the structured decision tree.
        """
        llm_output = sanitize_string(llm_output).expandtabs(4)

        # The text needs a newline at the end for the indentation logic to work correctly.
        llm_output = llm_output.strip() + "\n"

        # 1. Parse the text into a tree object
        parse_tree = self.parser.parse(llm_output)

        # 2. Transform the tree into our final dictionary
        structured_tree = self.transformer.transform(parse_tree)

        return structured_tree


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
    
    print(repr(llm_output_text_simple))
    llm_output_text_simple = "\nDECISION POINT: Is valproic acid applicable?\n\tIF 'Yes':\n\t\tOUTCOME: Prescribe valproic acid.\n\tIF 'No':\n\t\tOUTCOME: Refer to neurologist.\n"
    print(repr(llm_output_text_simple))

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

    print(repr(decision_tree_grammar))
    print(repr(sanitize_string(decision_tree_grammar)))

    # Create a parser and process the text
    parser = LLMTreeParser()
    decision_tree = parser.parse(llm_output_text_simple)

    # Print the resulting structured dictionary
    print(json.dumps(decision_tree, indent=4))

test()


def parser_util(llm_output: str) -> Dict[Any, Any]:
    """
    Utility function to parse LLM output into a structured format.
    """
    parser = LLMTreeParser()
    return parser.parse(llm_output)
