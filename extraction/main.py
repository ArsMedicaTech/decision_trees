"""
Decision tree extraction from medical text.
"""
from extraction.tree_parser import parser_util
from extraction.pipeline import get_response, load_topic_map



if __name__ == "__main__":
    response = get_response()
    print(response)

    topic_map = load_topic_map()
    
    url = topic_map.get('High Blood Pressure')
    #html_content = fetch_html_via_url(url)
    #response = extract_from_html(html_content)
    #print(response)

    parsed_data = parser_util(response)
    print(parsed_data)

# Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers
# pip install xformers.
