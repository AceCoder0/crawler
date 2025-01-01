# # import sys
# # from pathlib import Path
# # print(Path(__file__).parents[1])
# sys.path.append(Path(__file__).parents[1])
# from parsers.html_similarity.lm_path import lm_path
from lm_path import lm_path
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from typing import List
import numpy as np

def embed_html(htmls: List[str], model_name='BAAI/bge-large-zh-v1.5') -> np.ndarray:
    """
    Generate embeddings for a list of HTML strings.

    Parameters:
    htmls (List[str]): A list of HTML strings.
    model_name (str): The name of the language model to use.

    Returns:
    np.ndarray: A 2D NumPy array where each row is the embedding of an HTML string.
    """
    # Extract text from HTML strings
    def extract_text(html_str):
        soup = BeautifulSoup(html_str, 'html.parser')
        return soup.get_text()

    # Load tokenizer and model
    model_path = lm_path[model_name]  # Ensure lm_path is defined
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode

    # Generate embeddings for each HTML string
    embeddings = []
    for html in htmls:
        # Extract text from HTML
        text = extract_text(html)

        # Tokenize the text
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the pooler output as the sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(embedding)

    # Convert list of embeddings to a NumPy array
    return np.array(embeddings)

def semantic_distance(html_str1: str, html_str2: str, model_name='BAAI/bge-large-zh-v1.5') -> float:
    """
    Compute the semantic distance between two HTML strings.
    
    Parameters:
    html_str1 (str): The first HTML string.
    html_str2 (str): The second HTML string.
    model_name (str): The name of the language model to use.
    
    Returns:
    float: The semantic distance between the two HTML strings.
    """
    # Extract text from HTML strings
    def extract_text(html_str):
        soup = BeautifulSoup(html_str, 'html.parser')
        return soup.get_text()
    
    # text1 = extract_text(html_str1)
    # text2 = extract_text(html_str2)
    text1 = html_str1
    text2 = html_str2
    model_path = lm_path[model_name]
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    # Tokenize the texts
    inputs1 = tokenizer([text1], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    inputs2 = tokenizer([text2], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    
    # Get embeddings
    with torch.no_grad():
        embeddings1 = model(**inputs1)
        embeddings2 = model(**inputs2)
    
    # Use the pooler output for the sentence embedding
    embedding1 = embeddings1[0][:, 0].numpy()
    embedding2 = embeddings2[0][:, 0].numpy()
    
    # Compute cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    # Compute semantic distance
    distance = 1.0 - similarity
    
    return distance

def test_semantic_distance():
    """
    Test the semantic_distance function.
    """
    # Test case 1: Identical HTML strings
    html_str1 = "<html><body><p>Hello, world!</p></body></html>"
    html_str2 = "<html><body><p>Hello, world!</p></body></html>"
    distance = semantic_distance(html_str1, html_str2)
    print(f"Test case 1 - Identical HTML strings: Distance = {distance}")
    assert distance < 1e-5, "Test case 1 failed: Distance should be 0.0 for identical strings."

    # Test case 2: Similar HTML strings
    html_str1 = "<html><body><p>Hello, world!</p></body></html>"
    html_str2 = "<html><body><p>Hi, world!</p></body></html>"
    distance = semantic_distance(html_str1, html_str2)
    print(f"Test case 2 - Similar HTML strings: Distance = {distance}")
    assert 0.0 < distance < 1.0, "Test case 2 failed: Distance should be between 0.0 and 1.0 for similar strings."

    # Test case 3: Completely different HTML strings
    html_str1 = "<html><body><p>Hello, world!</p></body></html>"
    html_str2 = "<html><body><p>Goodbye, world!</p></body></html>"
    distance = semantic_distance(html_str1, html_str2)
    print(f"Test case 3 - Completely different HTML strings: Distance = {distance}")
    assert distance > 0.5, "Test case 3 failed: Distance should be greater than 0.5 for completely different strings."

    # Test case 4: Empty HTML strings
    html_str1 = ""
    html_str2 = ""
    distance = semantic_distance(html_str1, html_str2)
    print(f"Test case 4 - Empty HTML strings: Distance = {distance}")
    assert distance == 0.0, "Test case 4 failed: Distance should be 0.0 for empty strings."

    # Test case 5: One empty and one non-empty HTML string
    html_str1 = "<html><body><p>Hello, world!</p></body></html>"
    html_str2 = ""
    distance = semantic_distance(html_str1, html_str2)
    print(f"Test case 5 - One empty and one non-empty HTML string: Distance = {distance}")
    assert distance == 1.0, "Test case 5 failed: Distance should be 1.0 for one empty and one non-empty string."

    print("All test cases passed!")

if __name__ == '__main__':
    test_semantic_distance()