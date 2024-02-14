# PDF Reader
## Table of Contents
- [PDF Reader](#pdfreader)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Features](#features)
  - [Setup](#setup)
    - [Requirements](#requirements)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Starting the Chatbot](#starting-the-chatbot)
    - [Interacting with the Chatbot](#interacting-with-the-chatbot)
  - [Acknowledgments](#acknowledgments)

## Description

This project levarages OpenAI embeddings and Langchain to train OpenAI's GPT model on your own PDF files. This allows us to ask questions about our own data and allow the large language model to analyse and answer us.

## Features

- Train ChatGPT's AI model on your own documents in seconds.
- Extract salient information easily by asking questions.
- Ask for summaries or specific information. 

## Setup

### Requirements

- Python 3.8+
- Pip package installer for Python
- Internet connection for downloading dependencies

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/chonguschonguschongus/PDF-reader.git
   ```

2. Navigate to project directory
   ```bash
   cd PDF-reader
   ```

3. Install the required dependencies:
   ```bash
   pip install streamlit pypdf2 langchain faiss-cpu         
   ```

4. [Obtain your OpenAI API key](https://platform.openai.com/docs/quickstart?context=python)

5. Insert your OpenAI API key
   ```python
   OPENAI_API_KEY = "YOUR_KEY_HERE" 
   ```

## Usage
### Starting the PDF reader
Open terminal and run `streamlit run pdfreader.py`

### Uploading your own documents
Click the "Browse files" button to browse locally and select your desired document.

![browse button](https://github.com/chonguschonguschongus/PDF-reader/blob/main/images/browse%20files.png)

### Ask away!
Ask any questions you might have and the chatbot will reply you with information from your document!

![example image](https://github.com/chonguschonguschongus/PDF-reader/blob/main/images/example.png)


## Acknowledgements 
-  [OpenAI](https://openai.com/product) for providing the GPT model.
-  Python Software Foundation for the Python programming language.



