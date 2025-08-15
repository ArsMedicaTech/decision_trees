"""
Decision tree extraction from medical text.
"""
from typing import Dict

from pipeline import get_response


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
