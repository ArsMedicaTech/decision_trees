"""
This module contains utility functions for parsing JSON decision trees
and converting them into a format suitable for decision-making logic.

Meant for use in the `ArsMedicaTech` application, but can be adapted for other uses.
"""
from typing import Dict, Any, Union

def parse_condition_key(key: str) -> Union[tuple, str]:
    """
    Converts a string condition key from JSON into a Python condition tuple.
    """
    key = key.strip()
    if '-' in key:
        lower, upper = map(int, key.split('-'))
        return ('in', range(lower, upper + 1))
    elif key.startswith('>='):
        return ('>=', int(key[2:].strip()))
    elif key.startswith('<'):
        return ('<', int(key[1:].strip()))
    else:
        return key  # Fallback for unknown format


def parse_json_to_python(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively parses a JSON decision tree dictionary and converts
    string condition keys into Python tuples.
    """
    parsed = {
        "question": json_data["question"],
        "branches": {}
    }

    for key, value in json_data.get("branches", {}).items():
        condition = parse_condition_key(key)

        if isinstance(value, dict):
            parsed["branches"][condition] = parse_json_to_python(value)
        else:
            parsed["branches"][condition] = value

    return parsed
