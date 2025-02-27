import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import tempfile
import os
import dotenv

dotenv.load_dotenv()

# Set the page configuration
st.set_page_config(
    page_title="Medical ChatBot Assistant",
    page_icon="🩺",
    layout="wide"
)

# Initialize session state for storing documents and questions
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'questions' not in st.session_state:
    st.session_state.questions = [{"question": "", "answered": False}]

# Add a title and description
st.title("🩺 Medical ChatBot Assistant")
st.write("- Upload medical documents and ask multiple questions about their content. The file will take 2 to 3 minutes to process, after which the chatbot will be ready to use.")
st.write("- Refreshing the page will remove all your questions, answers, and the uploaded file.")
# Sidebar for PDF upload
with st.sidebar:
    st.header("📤 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload your medical PDFs",
        type=['pdf'],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} document(s) uploaded")

def format_response(response_text):
    """Format the response in a more readable way"""
    # Split response into sections
    sections = response_text.split('\n\n')
    formatted_sections = []
    
    for section in sections:
        if ':' in section:
            # Handle sections with titles
            title, content = section.split(':', 1)
            formatted_sections.append(f"### {title.strip()}")
            
            # Format bullet points
            points = content.strip().split('\n')
            for point in points:
                if point.strip():
                    formatted_sections.append(f"- {point.strip()}")
        else:
            # Handle plain paragraphs
            formatted_sections.append(section)
    
    return '\n'.join(formatted_sections)

def process_pdfs(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load the PDF
        loader = PyPDFLoader(tmp_file_path)
        documents.extend(loader.load())
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

# Main content area
if uploaded_files and not st.session_state.processed_pdfs:
    with st.spinner("Processing documents..."):
        try:
            vectorstore = process_pdfs(uploaded_files)
            st.session_state.vectorstore = vectorstore
            st.session_state.processed_pdfs = True
            st.success("✅ Documents processed successfully!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

# User input section
with st.container():
    st.markdown("### Ask Your Questions")
    
    # Function to add new question
    def add_question():
        st.session_state.questions.append({"question": "", "answered": False})
    
    # Function to clear all questions
    def clear_questions():
        st.session_state.questions = [{"question": "", "answered": False}]
    
    # Create columns for control buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("➕ Add Question"):
            add_question()
    with col2:
        if st.button("🗑️ Clear All"):
            clear_questions()
    
    # Display all question input fields
    for idx, q in enumerate(st.session_state.questions):
        col1, col2 = st.columns([8, 1])
        with col1:
            st.session_state.questions[idx]["question"] = st.text_area(
                f"Question {idx + 1}:",
                value=q["question"],
                placeholder="Example: What are the symptoms of diabetes?",
                height=100,
                key=f"question_{idx}"
            )
        with col2:
            if st.button("❌", key=f"remove_{idx}", help="Remove this question"):
                st.session_state.questions.pop(idx)
                st.rerun()
    
    submit_button = st.button("🔍 Get Answers", type="primary")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

def generate_response(query, vectorstore):
    # Search for relevant documents
    relevant_docs = vectorstore.similarity_search(query, k=3)
    
    # Create a context-aware prompt
    prompt = f"""You are a knowledgeable medical assistant. Using the provided medical document context, answer the user's questions accurately and professionally.
    Format your response in a clear and structured manner using:
    ● Bullet points (●) for lists, with each point on a separate line
    ● Clear section headings followed by a colon (:)
    ● Short, concise sentences
    If the information is not available in the documents, clearly state that you cannot provide an answer based on the given content.
    The response should be well-formatted and without using asterisks (*) with each bullet point displayed on a new line for better readability.
    {[doc.page_content for doc in relevant_docs]}

    User Question: {query}

    Please provide a clear, accurate, and well-structured answer based on the medical documents:"""
    
    # Generate response
    response = llm.invoke(prompt)
    return format_response(response.content)

# Process questions and generate responses
if submit_button:
    if not uploaded_files:
        st.warning("⚠️ Please upload some medical documents first.")
    else:
        # Filter out empty questions
        valid_questions = [q for q in st.session_state.questions if q["question"].strip()]
        
        if not valid_questions:
            st.warning("⚠️ Please enter at least one question before submitting.")
        else:
            with st.spinner("Analyzing documents and generating answers..."):
                try:
                    for q in valid_questions:
                        if not q["answered"]:
                            response = generate_response(q["question"], st.session_state.vectorstore)
                            
                            # Create an expander for each Q&A pair
                            with st.expander(f"Q: {q['question'][:100]}...", expanded=True):
                                st.markdown(response)
                            
                            # Store in chat history
                            st.session_state.chat_history.append({
                                "question": q["question"],
                                "answer": response
                            })
                            q["answered"] = True
                    
                    st.success("✅ All responses generated successfully!")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# Display chat history with improvements
if st.session_state.chat_history:
    st.markdown("### 📚 Previous Questions and Answers")
    for i, qa in enumerate(reversed(st.session_state.chat_history[-10:]), 1):
        with st.expander(f"Q{len(st.session_state.chat_history)-i+1}: {qa['question'][:50]}...", expanded=False):
            st.markdown(qa['answer'])

# Add helpful tips
with st.expander("💡 Tips for asking questions"):
    st.markdown("""
    - Be specific in your medical questions.
    - Include relevant symptoms or conditions.
    - Ask multiple questions at a time.
    - Provide context when necessary.
    """)

# Footer
st.markdown("---")
st.markdown("*Note: This is an AI assistant for informational purposes only. Always consult with healthcare professionals for medical advice.*")
