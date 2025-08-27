"""
Decision tree extraction from medical text.
"""
from extraction.chunking import run_extraction_pipeline
from extraction.pipeline import extract_from_html, fetch_html_from_local_file, fetch_html_via_url, load_topic_map


def write_decision_tree_to_file(decision_tree, file_path):
    with open(file_path, 'w') as f:
        f.write(decision_tree)


def main_single_article(article_name: str, debug: bool = False):
    """
    Process a single article to extract its decision tree.

    NOTE: This assumes a mapping of one decision tree per article.
    We may want to explore the possibility of multiple distinct trees per article,
    or even trees that span multiple articles.
    """
    topic_map = load_topic_map()
    url = topic_map.get(article_name)

    if not url:
        print("No URL found for the specified topic.")
        exit(1)

    # 1. Fetch the article
    #html_content = fetch_html_via_url(url)

    # Using locally saved article instead...
    u = url.split('/')
    doi = u[-2] + '_' + u[-1]
    file_path = f"extraction/articles/{doi}.htm"
    html_content = fetch_html_from_local_file(file_path)

    # 2. Extract clean text from the HTML
    article_text = extract_from_html(html_content)

    print("[DEBUG] Extracted article text length:", len(article_text))

    # 3. Run the full extraction pipeline
    final_decision_tree = run_extraction_pipeline(article_text, debug=debug)

    print("\n\n--- FINAL MERGED DECISION TREE ---")
    print(final_decision_tree)

    write_decision_tree_to_file(final_decision_tree, f"decision_trees/{article_name}_tree.txt")


def main_all():
    topic_map = load_topic_map()

    article_names = list(topic_map.keys())

    for article_name in article_names:
        main_single_article(article_name)


if __name__ == "__main__":
    main_single_article("High Blood Pressure", debug=True)



# Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers
# pip install xformers.
