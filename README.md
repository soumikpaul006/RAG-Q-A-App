# RAG Q&A Application

A Streamlit-based application that implements Retrieval-Augmented Generation (RAG) for question-answering across multiple document formats. The application uses FAISS for efficient similarity search, HuggingFace embeddings, and LLaMA 3 for generating responses.

## Features

- Multiple input formats supported:
  - Web links (URLs)
  - PDF documents
  - Plain text
  - Word documents (DOCX)
  - Text files (TXT)
- Vector similarity search using FAISS
- Document embedding using HuggingFace's sentence transformers
- Question answering using Meta's LLaMA 3 model
- User-friendly interface built with Streamlit

## Prerequisites

- Python 3.8+
- HuggingFace API key

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install the required dependencies:
```bash
pip install streamlit faiss-cpu python-docx PyPDF2 langchain langchain-community langchain-huggingface
```

3. Create a `secret_api_keys.py` file in the project root:
```python
huggingface_api_key = "your-huggingface-api-key"
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Select your input type from the dropdown menu:
   - For web links: Enter the number of links and paste URLs in the sidebar
   - For documents: Upload your PDF, DOCX, or TXT file
   - For text: Directly paste your text in the input field

3. Click "Proceed" to process your input and create the vector store

4. Enter your question in the text input field and click "Submit" to get an answer

## Technical Details

### Document Processing Pipeline

1. **Input Processing**: Documents are loaded and converted to text based on their format
2. **Text Splitting**: Content is split into smaller chunks using LangChain's CharacterTextSplitter
3. **Embedding**: Text chunks are embedded using HuggingFace's sentence-transformers
4. **Vector Store**: Embeddings are stored in a FAISS index for efficient retrieval
5. **Question Answering**: Uses LLaMA 3 to generate responses based on retrieved relevant context

### Key Components

- **Embeddings**: Uses the "sentence-transformers/all-mpnet-base-v2" model
- **Vector Store**: FAISS with L2 distance metric for similarity search
- **LLM**: Meta-Llama-3-8B-Instruct model through HuggingFace's API
- **UI**: Streamlit for the web interface

## Configuration

- Chunk size: 1000 characters
- Chunk overlap: 100 characters
- LLM temperature: 0.6 (controls response creativity)
- FAISS index: FlatL2 (exact search with L2 distance)

## Error Handling

The application includes error handling for:
- Invalid input formats
- File processing errors
- API communication issues
- Query processing errors

## Limitations

- Web link processing is limited to 20 URLs
- Performance may vary based on input size and available computational resources
- Requires active internet connection for API calls
- API rate limits apply based on HuggingFace account type

## Security Considerations

- API keys should be kept secure and never committed to version control
- Input validation is implemented for all file uploads
- In-memory document store is used for data privacy


 ## Local URL: http://localhost:8501
 ## Network URL: http://192.168.68.91:8501


![image](https://github.com/user-attachments/assets/595f7df0-0385-47f8-a59f-cc019d0d3409)


![image](https://github.com/user-attachments/assets/9ce57155-5223-44e3-9684-3476022a21b0)


![image](https://github.com/user-attachments/assets/f8919295-2f0f-46fe-8252-d735bffee262)


![image](https://github.com/user-attachments/assets/f08b5cca-4ba5-49e8-b313-0dbb3891cbdc)






![image](https://github.com/user-attachments/assets/8b0a5df2-ae91-46c9-a63e-b95fd56a4b6f)
