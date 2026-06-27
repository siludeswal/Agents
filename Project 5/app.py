import streamlit as st
from langchain_core.messages import HumanMessage
from model import graph

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NewsGenie",
    page_icon="🗞️",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e8e8e8;
}

/* Header */
.genie-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.genie-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #f5c542;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.genie-header p {
    color: #888;
    font-size: 0.95rem;
}

/* Chat bubbles */
.bubble-user {
    background: #1e2130;
    border-left: 3px solid #f5c542;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
}
.bubble-bot {
    background: #161b27;
    border-left: 3px solid #4e9af1;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 6px;
}
.badge-user  { background: #f5c54222; color: #f5c542; }
.badge-bot   { background: #4e9af122; color: #4e9af1; }

/* Input area */
div[data-testid="stChatInput"] textarea {
    background: #1a1f2e !important;
    color: #e8e8e8 !important;
    border: 1px solid #2e3347 !important;
    border-radius: 10px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1017;
    border-right: 1px solid #1e2130;
}
.sidebar-label {
    font-size: 0.75rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

/* Clear button */
.stButton > button {
    width: 100%;
    background: #1e2130;
    color: #f5c542;
    border: 1px solid #f5c54255;
    border-radius: 8px;
    font-size: 0.85rem;
    padding: 0.45rem;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #f5c54222;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # list of {"role", "content", "category"}

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session_1"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🗞️ NewsGenie")
    st.markdown("---")

    st.markdown('<p class="sidebar-label">Session ID</p>', unsafe_allow_html=True)
    thread_id = st.text_input(
        label="thread_id",
        value=st.session_state.thread_id,
        label_visibility="collapsed",
    )
    if thread_id != st.session_state.thread_id:
        st.session_state.thread_id = thread_id
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown('<p class="sidebar-label">What I can do</p>', unsafe_allow_html=True)
    st.markdown("""
- 💬 **General chat** — greetings, coding, AI  
- 📰 **Latest news** — current affairs & headlines  
- 🔍 **Fact check** — verify claims & statements
""")
    st.markdown("---")

    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="genie-header">
    <h1>🗞️ NewsGenie</h1>
    <p>Ask me anything — news, facts, or just chat.</p>
</div>
""", unsafe_allow_html=True)

# ── Chat History ──────────────────────────────────────────────────────────────

CATEGORY_EMOJI = {"general": "💬", "news": "📰", "fact": "🔍"}

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="bubble-user">
            <span class="badge badge-user">You</span><br>{msg["content"]}
        </div>""", unsafe_allow_html=True)
    else:
        emoji = CATEGORY_EMOJI.get(msg.get("category", ""), "🤖")
        st.markdown(f"""
        <div class="bubble-bot">
            <span class="badge badge-bot">{emoji} NewsGenie</span><br>{msg["content"]}
        </div>""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask about news, verify a fact, or just say hi…")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Invoke graph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.spinner("Thinking…"):
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

    answer   = result["final_result"]
    category = result.get("category", "general")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "category": category,
    })

    st.rerun()