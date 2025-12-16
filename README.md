# Natural Language Classifier with LLM-powered Chatbot

An end-to-end machine learning project that combines a traditional NLP classifier
with a large language model (LLM) to provide an interactive chatbot experience.

This project demonstrates how to build a complete AI pipeline: from data ingestion
and preprocessing, to model training and inference, and finally to human-friendly
interaction using LLM-powered reasoning and Retrieval-Augmented Generation (RAG).

---

## Project Overview

This repository implements a full natural language processing workflow:

1. **Data Ingestion & Preprocessing**

   - Clean and prepare a real-world text dataset
   - Persist processed data locally for reuse
   - Designed to be extendable to cloud-based pipelines

2. **Model Training**

   - Train a machine learning classifier for natural language understanding
   - Save the trained model, vectorizer, and label mappings for inference

3. **Classifier Interface**

   - Interact with the trained model through a terminal-based interface
   - Return predictions based on user input

4. **LLM-powered Chatbot (RAG)**
   - Use a large language model to interpret user input
   - Enrich classifier predictions with natural language explanations
   - Implement Retrieval-Augmented Generation (RAG) using local knowledge files

---

## Tech Stack

- Python
- Machine Learning (NLP classification)
- Scikit-learn
- Large Language Models (LLM)
- Retrieval-Augmented Generation (RAG)
- Command-line Interface (CLI)

---

## 🛠 Usage

### 0. Set the nesscery environment

Run the script to set environment:

```bash
python -m venv venv
source venv/bin/activate
export PYTHONPATH=$(pwd)
pip install --upgrade pip
pip install -r requirements.txt
```

### 1. Preprocess the dataset

Run the ingestion script to clean and prepare the data:

```bash
python ingest.py
```

### 2. Train the model

Run the train model script to train and save the model, vectorizer, mapping:

```bash
python train_model.py
```

### 3.Talk with the model

Simple run simple_interface.py and you can start chat with model in terminal.

```bash
python simple_interface.py
```

### 3.Talk with the model with llm

If you want use natrual language talk with model, simple run chatbot_interface.py! It will using llm to analyse input and polish output

```bash
python chatbot_interface.py
```
