"""
Pipeline helper functions for long-document extraction.
"""
from typing import List

from extraction.model import construct_model
from extraction.prompt import build_prompt, Message


# --- Step 1: Chunking Function ---
def chunk_text(full_text: str) -> List[str]:
    """
    Splits the full text into smaller, paragraph-based chunks.
    """
    # Split by double newlines, which often delineate paragraphs
    paragraphs = full_text.split('\n\n')
    # Filter out any empty or whitespace-only paragraphs
    return [p.strip() for p in paragraphs if p.strip()]


# --- Step 2: Processing (Map) Function ---
def process_chunk(text_chunk: str, model, tokenizer, generation_config) -> str:
    """
    Runs the extraction prompt on a single chunk of text.
    """
    print(f"--- Processing Chunk (length: {len(text_chunk)} chars) ---")
    messages = build_prompt(medical_text=text_chunk)
    
    # Call the built-in chat method on the single chunk
    response_text = model.chat(tokenizer, messages, generation_config=generation_config)
    
    # Simple filter to check if the model found a tree
    if "DECISION POINT" not in response_text and "OUTCOME" not in response_text:
        print("--> No decision tree found in this chunk.")
        return ""

    print("--> Found partial decision tree.")
    return response_text


# --- Step 3: Synthesizing (Reduce) Function ---
def synthesize_trees(partial_trees: List[str], model, tokenizer, generation_config) -> str:
    """
    Merges a list of partial decision trees into a single, coherent tree.
    """
    print("\n--- Synthesizing all partial trees ---")
    
    # Join all the partial tree strings into one block
    all_partials = "\n---\n".join(partial_trees)

    # A new prompt specifically for the synthesis task
    synthesizer_prompt = f"""
    You are an expert at consolidating and merging multiple, partial medical decision trees 
    into a single, comprehensive version.

    Here are the partial decision trees extracted from different sections of a single medical article:
    {all_partials}

    YOUR TASK:
    Combine the logic from these partial trees into one single, logically correct, and 
    de-duplicated master decision tree. Connect related branches, resolve any redundancies,
    and ensure the final output is a single, coherent tree.
    """
    
    messages: List[Message] = [
        Message(role="user", content=synthesizer_prompt)
    ]
    
    final_tree = model.chat(tokenizer, messages, generation_config=generation_config)
    return final_tree


# --- Main Orchestrator ---
def run_extraction_pipeline(full_text: str) -> str:
    """
    Orchestrates the full chunk -> process -> synthesize pipeline.
    """
    # Load the model and tokenizer once for the entire pipeline
    model, tokenizer, generation_config = construct_model()
    if model is None:
        return "Model construction failed."

    # 1. Chunk the document
    print("[DEBUG] Chunking document...")
    text_chunks = chunk_text(full_text)
    
    # 2. Process each chunk to get partial trees
    partial_trees = []
    for chunk in text_chunks:
        print(f"[DEBUG] Processing chunk (length: {len(chunk)} chars)")
        result = process_chunk(chunk, model, tokenizer, generation_config)
        if result:
            partial_trees.append(result)
            
    if not partial_trees:
        return "No decision trees could be extracted from the document."
    
    # 3. Synthesize the partial trees into a final result
    final_result = synthesize_trees(partial_trees, model, tokenizer, generation_config)
    
    return final_result

