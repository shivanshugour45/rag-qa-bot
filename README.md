# RAG QA Chatbot

This is a simple QA (Question Answering) Chatbot implemented in Python using various libraries and frameworks.

## Overview

This chatbot allows users to upload PDF or document files and ask questions related to the content of those files. The chatbot then provides answers to those questions based on the information extracted from the uploaded files.

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/yourusername/qa-chatbot.git
    ```

2. Navigate to the project directory:

    ```bash
    cd qa-chatbot
    ```

3. Create a python virtual environment.

    ```bash
    python -m venv .venv
    ```

4. Activate the python virtual environemnt:

    ```bash
    .venv\\scripts\\Activate
    ```

5. Install the required dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

6. Setup environment variables:
    Since this chatbot uses Google Gemini Model as the underlying llm, in order to run this file create a .env file inside the .venv folder and add `GOOGLE_API_KEY` environment variable in it.

## Usage

To run the chatbot, execute the `app.py` script:

```bash
streamlit run app.py
```

The chatbot will be accessible through a web interface provided by Streamlit. Users can upload PDF or document files, ask questions related to the content of those files, and receive answers from the chatbot.

## Features

- **File Upload**: Users can upload PDF or document files containing text content.
- **Question Answering**: Users can ask questions related to the content of the uploaded files.
- **Interactive Chat Interface**: The chatbot provides an interactive interface for users to ask questions and receive answers.
- **Reset and Stop**: Users can reset the chat session or stop the chatbot at any time.

## File Structure

- **app.py**: Contains the main script to run the chatbot.
- **datavectoriser.py**: Contains script for creating vectorstores.
- **README.md**: Provides information about the project.
- **requirements.txt**: Lists all the Python dependencies required for the project.
- **data/**: Directory to store uploaded files.
- **vectorstore/db_lancedb**: Directory to store vector embeddings generated from uploaded files.

## Dependencies

- **Python**: The programming language used for development.
- **Streamlit**: A web application framework used to create the chatbot interface.
- **PyPDF2**: A library for reading PDF files.
- **python-docx**: A library for reading and writing Word documents.
- **dotenv**: A library for loading environment variables from a .env file.
- **langchain**: A library for natural language processing tasks.
- **langchain_google_genai**: A library for integrating Google's Generative AI models with langchain.
- **LanceDB**: A library for managing vector embeddings.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- Shivanshu Gour (@shivanshugour45)