"""
Decision tree extraction from medical text.
"""
from extraction.chunking import run_extraction_pipeline
from extraction.tree_parser import parser_util
from extraction.pipeline import extract_from_html, fetch_html_via_url, get_response, load_topic_map


if __name__ == "__main__":
    response = get_response()
    print(response)

    topic_map = load_topic_map()
    
    url = topic_map.get('High Blood Pressure')

    if not url:
        print("No URL found for the specified topic.")
        exit(1)

    # 1. Fetch the article
    html_content = fetch_html_via_url(url)

    # 2. Extract clean text from the HTML
    article_text = extract_from_html(html_content)

    # 3. Run the full extraction pipeline
    final_decision_tree = run_extraction_pipeline(article_text)

    print("\n\n--- FINAL MERGED DECISION TREE ---")
    print(final_decision_tree)

    parsed_data = parser_util(response)
    print(parsed_data)

# Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers
# pip install xformers.
