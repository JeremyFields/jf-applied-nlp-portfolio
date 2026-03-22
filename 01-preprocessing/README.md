# NLP Portfolio: Pre-processing Pipeline

## Overview
This project demonstrates a professional NLP pre-processing pipeline that transforms raw HTML data into a structured, machine-learning-ready format using **spaCy**, **BeautifulSoup**, and **Pandas**.

## Setup & Installation
1.  **Clone the repository and navigate to this folder:**
    ```bash
    git clone https://github.com/JeremyFields/jf-applied-nlp-portfolio.git
    cd jf-applied-nlp-portfolio/01-preprocessing/
    ```
2.  **Activate venv and install dependencies using `uv` (or `pip`):**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -r ./requirements.txt --link-mode copy
    ```
3.  **Download the spaCy language model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Technical Workflow
1.  **Direct Text Acquisition:** 
    *   Targeted extraction of body content from HTML using **BeautifulSoup**.
    *   Focused on isolating article content by HTML attributes (e.g., `id="Contentplaceholder1_C011_Col00"`).
2.  **Heuristic Text Normalization:**
    *   Implemented cleaning logic to resolve "noisy" web text (extra whitespace, newline clusters).
    *   Applied **Regex** to ensure sentence-final punctuation, maintaining data integrity for the segmenter.
3.  **spaCy Pipeline Orchestration:**
    *   Executed a full English pipeline (`en_core_web_sm`) to generate a `Doc` object.
    *   Extracted high-value features: **Lemmas** (word roots), **Part-of-Speech (POS) tags**, and **Named Entities (NER)**.
4.  **Custom Tokenizer Engineering:**
    *   **The Problem:** Default tokenizers often fail on specific formats, such as slash-delimited dates (`06/01/2023`), treating them as a single token.
    *   **The Solution:** Extended the spaCy tokenizer by modifying **Infix Rules**. Added a custom regex pattern to ensure correct segmentation of dates into constituent sub-tokens while preserving non-destructive offsets.
5.  **Structured Data Export:**
    *   Mapped the resulting `Span` and `Token` objects into a **Pandas DataFrame**, creating a granular, indexed dataset for downstream analysis.

## Tools & Libraries
*   **Python 3.x**
    *   **spaCy:** Pipeline management and custom component engineering.
    *   **BeautifulSoup4:** Web scraping and HTML parsing.
    *   **Pandas:** Data structuring and tabular exploration.
    *   **Regex (re):** Pattern-based text cleaning.