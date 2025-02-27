# Medical ChatBot Assistant

## Project Image
![Agentic AI CustomGPT](Medical_chatbot_Assistant.png)

## Overview
The **Medical ChatBot Assistant** is a Streamlit-based application that allows users to upload medical PDF documents and ask questions about their content. The chatbot uses Google's Gemini AI and FAISS for retrieval-augmented generation (RAG), enabling accurate and context-aware responses.

## Features
- 📂 **Upload multiple medical PDFs** for document analysis.
- 🔍 **Ask multiple questions** about the uploaded documents.
- 🤖 **AI-powered responses** using Google Gemini AI.
- 📑 **Automatic document processing** and text chunking.
- 🗄 **Vector-based search** for efficient information retrieval.
- 💾 **Session management** to store chat history and processed documents.
- ⏳ **Real-time processing** (takes 2-3 minutes per document).
- 📌 **Well-structured responses** with bullet points and headings.

## Tech Stack
- **Streamlit** (Frontend & UI framework)
- **LangChain** (Document processing and AI integration)
- **FAISS** (Vector store for semantic search)
- **Google Gemini AI** (LLM for response generation)
- **PyPDFLoader** (PDF document parsing)
- **Python** (Core language)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mfahadjbr/Medical_Chatbot_Assistant
   cd Medical_Chatbot_Assistant
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file and add the following:
     ```
     GEMINI_API_KEY=your_google_api_key
     ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```

## How to Use
1. **Upload PDFs** via the sidebar.
2. **Wait for processing** (2-3 minutes per document).
3. **Enter your medical-related questions** in the input fields.
4. **Click 'Get Answers'** to generate AI-powered responses.
5. **View previous Q&A history** or clear questions as needed.

## File Structure
```
medical-chatbot/
│── app.py                # Main Streamlit application
│── requirements.txt       # Required Python dependencies
│── .env                   # Environment variables (add manually)
│── README.md              # Project documentation
└── data/                  # (Optional) Store sample PDFs
```

## Limitations
- The chatbot provides **informational guidance only** and is **not a replacement for medical professionals**.
- Requires a stable internet connection for API calls.
- Large PDFs may take longer to process.

## License
This project is **open-source** under the MIT License.

## Contact
For any issues or suggestions, feel free to reach out:
- **Email**: mfahadjbr@gmail.com
- **GitHub**: [Your GitHub Profile](https://github.com/mfahadjbr)

---
*Note: Always consult with healthcare professionals for accurate medical advice.*

