import re
import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv

# Load environment variables
try:
    load_dotenv()  # For local development with .env file
except:
    pass  # If dotenv is not available

try:
    if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
        os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']
except:
    pass

# Configure Streamlit page
st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="chatbot_icon.png",
    layout="wide"
)

st.markdown(
    """
    <h1 style='display: flex; align-items: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/0/09/YouTube_full-color_icon_(2017).svg' width='40' style='margin-right:10px'/>
        YouTube Question & Answer Chatbot
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("Ask questions about any YouTube video based on its transcript!")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None
if 'video_loaded' not in st.session_state:
    st.session_state.video_loaded = False

def extract_video_id(url_or_id):
    """Extract video ID from YouTube URL or return the ID if it's already an ID"""
    if not url_or_id:
        return None
    
    # If it's already just the video ID (11 characters)
    if len(url_or_id) == 11 and url_or_id.isalnum():
        return url_or_id
    
    # Extract from various YouTube URL formats
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    
    return None

def get_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        trans_list = YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=['en'])
        transcript = ' '.join(i['text'].replace('\n', ' ') for i in trans_list)
        return transcript, None
    except TranscriptsDisabled:
        return None, 'Transcript is disabled for this video'
    except NoTranscriptFound:
        return None, 'Transcript is unavailable for this video'
    except Exception as e:
        return None, f'Error fetching transcript: {str(e)}'

def setup_qa_chain(transcript):
    """Set up the Q&A chain with the transcript"""
    # Split the transcript
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    chunks = splitter.create_documents([transcript])
    
    # Create vector store
    embedding = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
    vec_store = FAISS.from_documents(documents=chunks, embedding=embedding)
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model='models/gemini-2.5-pro')
    
    # Create retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vec_store.as_retriever(search_kwargs={'k': 2}),
        llm=llm
    )
    
    # Prompt template
    prompt_template = PromptTemplate(
        template="""Answer the following question from the provided context. If the context is insufficient, just say "I don't know based on the video transcript."\n
        Question: \n{ques}\n
        Context: \n{context}
        """
    )
    
    # Chain generation
    def context_gen(retriever_docs):
        context = ' '.join(doc.page_content for doc in retriever_docs)
        return context
    
    parser = StrOutputParser()
    
    chain_parallel = RunnableParallel({
        'ques': RunnablePassthrough(),
        'context': retriever | RunnableLambda(context_gen)
    })
    
    chain = chain_parallel | prompt_template | llm | parser
    
    return vec_store, retriever, chain

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📹 Video Setup")
    video_input = st.text_input(
        "Enter YouTube Video URL or Video ID:",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID or just VIDEO_ID",
        help="You can paste a full YouTube URL or just the video ID"
    )
    
    if st.button("Load Video", type="primary"):
        if video_input:
            video_id = extract_video_id(video_input)
            if video_id:
                with st.spinner("Loading video and setting up Q&A system..."):
                    transcript, error = get_transcript(video_id)
                    if transcript:
                        try:
                            vec_store, retriever, chain = setup_qa_chain(transcript)
                            st.session_state.vector_store = vec_store
                            st.session_state.retriever = retriever
                            st.session_state.chain = chain
                            st.session_state.current_video_id = video_id
                            st.session_state.video_loaded = True
                            st.success("Video loaded successfully!")
                            # st.info(f"📊 Transcript length: {len(transcript)} characters")
                            
                        except Exception as e:
                            st.error(f"Error setting up Q&A system: {str(e)}")
                    else:
                        st.error(f"{error}")
            else:
                st.error("Invalid YouTube URL or Video ID")
        else:
            st.error("Please enter a YouTube URL or Video ID")
    
    # Display video preview outside of button click
    if st.session_state.video_loaded and st.session_state.current_video_id:
        st.markdown("### 🎬 Video Preview")
        st.markdown(f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/{st.session_state.current_video_id}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

with col2:
    st.subheader("❓ Ask Questions")
    
    if st.session_state.chain is not None:
        st.success(f"Ready to answer questions about video: {st.session_state.current_video_id}")
        
        question = st.text_input(
            "Your Question:",
            placeholder="What is the main topic of this video?",
            help="Ask any question about the video content"
        )
        
        if st.button("Get Answer", type="secondary"):
            if question:
                with st.spinner("Generating answer..."):
                    try:
                        answer = st.session_state.chain.invoke(question)
                        st.markdown("### 💬 Answer:")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            else:
                st.warning("Please enter a question")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Quick question examples
        st.markdown("### 🔍 Example Questions:")
        example_questions = [
            "What is the main topic discussed?",
            "What are the key points mentioned?",
            "Can you summarize the video?",
            "What examples are given?"
        ]
        
        for i, example in enumerate(example_questions):
            if st.button(f"💡 {example}", key=f"example_{i}"):
                with st.spinner("Generating answer..."):
                    try:
                        answer = st.session_state.chain.invoke(example)
                        st.markdown("### 💬 Answer:")
                        st.markdown(answer)
                        # Add to chat history
                        st.session_state.chat_history.append({"question": example, "answer": answer})
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
    else:
        st.info("Please load a video first to start asking questions")

# Sidebar with instructions
with st.sidebar:
    st.markdown("## 📖 How to Use")
    st.markdown("""
    1. **Enter a YouTube URL or Video ID** in the input field
    2. **Click 'Load Video'** to process the video
    3. **Ask questions** about the video content
    4. **Get AI-powered answers** based on the transcript
    """)
    
    st.markdown("## 🔧 Features")
    st.markdown("""
    - **Multi-format support**: URLs or Video IDs
    - **Transcript language support**: English                
    - **Smart chunking**: Handles long transcripts
    - **Vector search**: Finds relevant content
    - **Context-aware**: Provides accurate answers
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain")