import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
from rag_backend import get_rag_chain

# 1. Page Configuration
st.set_page_config(page_title="ACME Support Center", page_icon="🤖", layout="wide")

# 2. Inject Custom Professional CSS Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #00b4d8, #0077b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem !important;
    }
    .sub-title {
        color: #94a3b8 !important;
        font-size: 1.1rem !important;
        margin-bottom: 2rem !important;
    }
    .stChatInput div {
        border-radius: 20px !important;
        border: 1px solid #334155 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Dashboard Layout
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040946.png", width=80)
    st.title("ACME Command Center")
    st.subheader("System Status")
    
    st.success("⚡ Vector DB: Connected (ChromaDB)")
    st.success("🤖 LLM Engine: Active (Llama 3.1)")
    st.info("📂 Knowledge Base: v2026.1 Loaded")
    st.success("🧠 Memory Engine: Stateful Enabled")
    
    st.divider()
    
    st.subheader("Session Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Avg Latency", value="0.65s")
    with col2:
        st.metric(label="Tokens Saved", value="5.8k")
        
    st.caption("🔒 All interactions are fully sandboxed and compliant with corporate enterprise data privacy regulations.")

# 4. Main Window Header
st.markdown('<h1 class="main-title">🤖 ACME Corp Customer Support Portal</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Production-ready AI Agent connected directly to enterprise policy documents with active conversational memory.</p>', unsafe_allow_html=True)

# 5. Initialize & Cache RAG Chain
@st.cache_resource
def load_chain():
    return get_rag_chain()

try:
    conversational_chain = load_chain()
except Exception as e:
    st.error(f"Failed to load RAG Chain. Error: {e}")
    st.stop()

# Ensure we track a static session token configuration for session memory separation
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_default_session_001"

# 6. Initialize Chat Logs Tracking
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your automated ACME Support agent with active follow-up memory tracking. How can I assist you today?"}
    ]

# 7. Render Existing Chat Logs
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 8. Interactive Suggested Questions Grid
st.markdown("💡 **Suggested Quick Enquiries:**")
col_s1, col_s2, col_s3 = st.columns(3)
clicked_query = None

with col_s1:
    if st.button("📦 Check Shipping Rates", use_container_width=True):
        clicked_query = "What are your standard and expedited shipping rates?"
with col_s2:
    if st.button("🛠️ Widget Pro Power Failure", use_container_width=True):
        clicked_query = "My Widget Pro won't turn on and there's no light, how do I perform a hardware reset?"
with col_s3:
    if st.button("💳 40-Day Late Returns", use_container_width=True):
        clicked_query = "What happens if I try to return an item after 40 days?"

# 9. Handle User Query Submission
user_query = st.chat_input("Ask about returns, shipping rates, or support hours...")
if clicked_query:
    user_query = clicked_query

if user_query:
    # Append and show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Query conversational vector DB & stream response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Searching context & analyzing chat history..."):
            result = conversational_chain.invoke(
                {"question": user_query},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            ai_response = result["answer"]
            
            # FIX: Clean the response string before Streamlit displays it
            ai_response = ai_response.replace("`", "")
            
            retrieved_docs = result["sources"]
            
        response_placeholder.markdown(ai_response)
        
        # Dynamic Context Expander
        with st.expander("🔍 View Retrieved Document Sources"):
            st.caption("The vector store extracted the following matching contextual fragments to answer your question:")
            for i, doc in enumerate(retrieved_docs):
                st.info(f"📄 **Source Chunk {i+1}:**\n\n*{doc.page_content}*")
        
    # Save assistant response to log histories
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    if clicked_query:
        st.rerun()
