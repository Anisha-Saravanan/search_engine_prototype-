# Search Engine Prototype

A Python-based **information retrieval system** that combines fundamental data structures with TF-IDF-based document ranking.

## Features

* **Trie** – Provides query autocomplete and prefix-based suggestions.
* **Inverted Index** – Enables efficient retrieval of documents containing query terms.
* **TF-IDF & Cosine Similarity** – Calculates document relevance to search queries.
* **AVL Tree** – Organizes and retrieves documents based on relevance scores.
* **Text Snippets** – Displays matching sections from retrieved documents.
* **Index Persistence** – Saves and loads the built search index.
* **CLI** – Provides an interactive interface for indexing and searching documents.

## Workflow

```text
Documents
   ↓
Tokenization
   ↓
Inverted Index + Trie
   ↓
TF-IDF & Cosine Similarity
   ↓
AVL Tree Ranking
   ↓
Ranked Search Results
```

## Technologies

* Python 3
* Trie
* Inverted Index
* AVL Tree
* TF-IDF
* Cosine Similarity

## Usage

```bash
python search_engine_prototype.py
```

The CLI allows you to build an index from `.txt` files, search documents, get autocomplete suggestions, save/load indexes, and view document snippets.
