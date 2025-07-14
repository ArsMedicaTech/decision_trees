import json

from utils import parse_json_to_python

# Example JSON decision tree for blood pressure
json_string = open('trees/blood_pressure.json', 'r').read()

# Parse the JSON string to Python dict
json_data = json.loads(json_string)

# Convert to decision tree format
parsed_tree = parse_json_to_python(json_data)

# Show result
import pprint
pprint.pprint(parsed_tree)
