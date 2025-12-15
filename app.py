



import streamlit as st
import os
from datetime import datetime
from notifier import notify_on_risks_simple

# Use HuggingFace embeddings (free, local)
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None
    print("HuggingFaceEmbeddings not available. Run: pip install langchain-community sentence-transformers")

from main import (
    read_contracts_from_pdfs,
    get_regulations_by_jurisdiction,
    detect_risks,
    apply_amendment_append,
    replace_clause_with_amendment,
    split_contract_text,
)

# Optional: LLM for chatbot (kept as user had it). If missing, chatbot will show a message.
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Compliance Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Global CSS ----------------
CSS = '''
<style>
:root {
    --accent1: #26c6da;    
    --accent2: #ff6ec7;    
    --accent3: #facc15;    
    --bg: #0f111a;         
    --card-bg: rgba(0, 0, 0, 0.35);  
    --button-gradient: linear-gradient(135deg, #ff6ec7 0%, #26c6da 100%);
    --text-light: #ffffff;  
    --text-muted: #cbd5e1;  
}

body, .stApp { 
    background: url("https://wallpapertag.com/wallpaper/full/b/c/4/143811-full-size-good-background-1920x1200-for-hd-1080p.jpg") no-repeat center center fixed;
    background-size: cover;
    font-family: 'Segoe UI', sans-serif; 
    color: var(--text-light);
}

section[aria-label="Sidebar"] {
  background: rgba(0,0,0,0.5);
  color: var(--text-light);
  font-weight: 500;
  backdrop-filter: blur(20px);
  transition: all 0.3s ease;
}
section[aria-label="Sidebar"] a, section[aria-label="Sidebar"] label, section[aria-label="Sidebar"] div { color: var(--text-light) !important; }

.topbar { display:flex; align-items:center; gap:12px; margin-bottom:20px; }
.app-title { 
    font-size:24px; 
    font-weight:700; 
    color: var(--accent1); 
    text-shadow: 0 0 10px var(--accent1), 0 0 20px var(--accent2);
    animation: neonGlow 2s ease-in-out infinite alternate;
}
.app-sub { color: var(--text-muted); font-style:italic; }

.card {
    background: var(--card-bg);
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    border-radius:20px; 
    padding:25px; 
    box-shadow: 0 0 20px rgba(38,198,218,0.5);
    transition: transform 0.3s, box-shadow 0.3s;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 35px rgba(255,110,199,0.8), 0 0 25px rgba(38,198,218,0.8);
    animation: pulseGlow 1.5s infinite alternate;
}

.metric { 
    font-size:34px; 
    font-weight:700; 
    background: -webkit-linear-gradient(45deg, var(--accent1), var(--accent2)); 
    -webkit-background-clip: text; 
    color: transparent; 
    text-shadow: 0 0 10px rgba(38,198,218,0.6);
    animation: neonGlow 2s ease-in-out infinite alternate;
}
.metric-label { color: var(--text-muted); font-size:14px; font-weight:600; }

.stButton>button {
    background: var(--button-gradient); 
    color: var(--text-light); 
    border-radius:12px; 
    padding:12px 20px; 
    font-weight:600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(255,110,199,0.4), 0 5px 15px rgba(38,198,218,0.4);
}
.stButton>button:hover { 
    transform: translateY(-3px); 
    box-shadow: 0 0 25px rgba(255,110,199,0.7), 0 0 25px rgba(38,198,218,0.7);
    animation: pulseGlow 1.5s infinite alternate;
}

.msg-user { 
    background: rgba(38,198,218,0.3); 
    border-radius:16px; 
    padding:14px; 
    margin:6px 0; 
    align-self:flex-end; 
    color: var(--text-light);
    box-shadow: 0 0 15px rgba(38,198,218,0.5);
    animation: pulseGlow 2s infinite alternate;
}
.msg-bot { 
    background: rgba(255,110,199,0.3); 
    border-radius:16px; 
    padding:14px; 
    margin:6px 0; 
    color: var(--text-light);
    box-shadow: 0 0 15px rgba(255,110,199,0.5);
    animation: pulseGlow 2s infinite alternate;
}
.chat-box { display:flex; flex-direction:column; gap:8px; }
.small-note { color: var(--text-muted); font-size:13px; }

@keyframes neonGlow {
    0% { text-shadow: 0 0 5px var(--accent1), 0 0 10px var(--accent2); }
    50% { text-shadow: 0 0 15px var(--accent1), 0 0 25px var(--accent2); }
    100% { text-shadow: 0 0 5px var(--accent1), 0 0 10px var(--accent2); }
}

/* Sidebar with green-yellow gradient */
section[aria-label="Sidebar"] {
  background: linear-gradient(135deg, #00ff00, #ffff00); /* green to yellow */
  color: var(--text-light);
  font-weight: 500;
  backdrop-filter: blur(20px);
  transition: all 0.3s ease;
}
section[aria-label="Sidebar"] a, 
section[aria-label="Sidebar"] label, 
section[aria-label="Sidebar"] div { 
    color: #000000 !important; /* text dark for contrast */
}


@keyframes pulseGlow {
    0% { box-shadow: 0 0 15px rgba(255,110,199,0.4), 0 0 15px rgba(38,198,218,0.4); }
    50% { box-shadow: 0 0 25px rgba(255,110,199,0.7), 0 0 25px rgba(38,198,218,0.7); }
    100% { box-shadow: 0 0 15px rgba(255,110,199,0.4), 0 0 15px rgba(38,198,218,0.4); }
}
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    logo_path = "data/contract.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=40)
    st.markdown("# 🔒 Compliance Studio")
    st.markdown("_Smart contract compliance & amendment assistant_")
    st.write("---")
    page = st.radio("Navigate", [
        "Dashboard", "Upload & Review", "Legal Chatbot", "Notifications",
    ])
    st.write("---")
    # st.markdown("<div class='small-note'>Logged in as: <b>Sonam Nayak</b></div>", unsafe_allow_html=True)

# ---------------- Load Contracts ----------------
contracts = read_contracts_from_pdfs() or []

# ---------------- Helper Functions ----------------

def compute_risky_contracts(jurisdiction="GLOBAL"):
    regs = get_regulations_by_jurisdiction(jurisdiction)
    risky = []
    for c in contracts:
        risks = detect_risks(c, regs)
        if risks:
            risky.append((c, risks))
    return risky


def render_contract_row(c, risks):
    st.markdown(f"""
    <div class='card' style="background-color:#0b1220;color:#fff;">
        <div style="font-weight:700;font-size:16px;margin-bottom:8px;">📄 {c['name']}</div>
        <div class='small-note' style="margin-bottom:8px;color:#cbd5e1;">Last modified: {c.get('last_modified','Unknown')}</div>
        <div style="color:#ddd;margin-bottom:8px;">{c['text'][:400] + ('...' if len(c['text']) > 400 else '')}</div>
        <div style="font-weight:600;color:#ffcc00;">Detected risks: {len(risks)}</div>
    </div>
    """, unsafe_allow_html=True)
    for r in risks[:4]:
        st.warning(f"{r.get('regulation','UNKNOWN')} — {r.get('sentence','')[:120]}...")

# ---------------- PAGE: Dashboard ---------------- 
if page == "Dashboard":
    st.markdown("<div class='card'><h2>🛡️ Compliance Studio — Dashboard</h2></div>", unsafe_allow_html=True)

    # ---- TOP SUMMARY CARDS (No Risk Analysis) ----
    total_contracts = len(contracts)
    need_revision = 0          # No risk analysis on dashboard
    # up_to_date removed because we don't need it anymore

    col1, col2 = st.columns([1,1])

    # ---------- BOX 1: Total Contracts ----------
    with col1:
        st.markdown(
            f"""
            <div class='card'>
                <div style='font-size:28px;font-weight:700'>📄 {total_contracts}</div>
                <div class='small-note'>Total Contracts</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------- BOX 3: Contracts For Review (Always 0) ----------
    with col2:
        st.markdown(
            f"""
            <div class='card'>
                <div style='font-size:28px;font-weight:700'>⚙️ {need_revision}</div>
                <div class='small-note'>Contracts For Review</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")


    # ---------------- OUR SERVICES ----------------
    st.markdown("## 🛠️ Our Services")
    st.markdown(
        """
        <div class='card' style='padding:20px'>
            <ul style='font-size:18px; line-height:1.7'>
                <li>📘 Automated Contract Text Extraction</li>
                <li>🔍 Regulation-Based Risk Detection</li>
                <li>✏️ AI-Powered Clause Amendment Suggestions</li>
                <li>📢 Real-Time Notifications for High-Risk Contracts</li>
                <li>🌍 Multi-Jurisdiction Compliance Checking</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------- HOW OUR SYSTEM WORKS ----------------
    st.markdown("## 🚀 How Our System Works", unsafe_allow_html=True)

    st.markdown(
    """
<div class='card' style='padding:25px; font-size:18px; line-height:1.8'>

<b>📤 1. Upload Contract</b><br>
Users upload PDF contracts to the system.<br><br>

<b>📑 2. Extract Clauses</b><br>
AI extracts and preprocesses contract text.<br><br>

<b>⚖️ 3. Compliance Check</b><br>
System compares clauses with regulations.<br><br>

<b>🚨 4. Risk & Violations Detection</b><br>
Any mismatch is flagged automatically.<br><br>

<b>✏️ 5. AI Amendment Suggestions</b><br>
System provides corrected clauses.<br><br>

<b>📬 6. Notifications Sent</b><br>
Email/SMS alerts for risky contracts.<br><br>

<b>🎉 7. Download Updated Contract</b><br>
User downloads the amended contract.

</div>
""",
    unsafe_allow_html=True
)


    # ---------------- WHY CHOOSE US ----------------
    st.markdown("## ⭐ Why Choose Compliance Studio?")
    st.markdown(
        """
        <div class='card' style='padding:20px'>
            <ul style='font-size:18px; line-height:1.7'>
                <li>⚡ 95% faster contract review process</li>
                <li>🔐 Secure & encrypted document handling</li>
                <li>📊 Smart AI-based risk detection</li>
                <li>🌍 Supports multiple jurisdictions</li>
                <li>🧠 Continuous AI learning for better accuracy</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- PAGE: Upload & Review ---------------- 
elif page == "Upload & Review":
    st.title("📤 Upload & 🛠️ Review Contracts")
    st.write("Upload a new contract and review detected risks in the same page.")

    # ----- Upload Section -----
    col1, col2 = st.columns(2)
    uploaded_contract = None

    with col1:
        uploaded_contract = st.file_uploader("Upload Contract PDF", type=['pdf'])
        if uploaded_contract:
            save_path = os.path.join('data','contracts', uploaded_contract.name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(uploaded_contract.getbuffer())
            st.success(f"Saved {uploaded_contract.name}")

    with col2:
        uploaded_compliance = st.file_uploader("Upload Compliance PDF", type=['pdf'])
        if uploaded_compliance:
            save_path = os.path.join('data','general', uploaded_compliance.name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(uploaded_compliance.getbuffer())
            st.success(f"Saved {uploaded_compliance.name}")

    st.write("---")

    # ----- Revision Manager Section -----
    all_contracts = read_contracts_from_pdfs() or []

    if not all_contracts:
        st.error("No contracts available for review.")
    else:
        names = ["Select contract"] + [c['name'] for c in all_contracts]
        sel = st.selectbox("Select contract to review", names)

        if sel != "Select contract":
            contract = next(c for c in all_contracts if c['name'] == sel)
            jurisdiction = st.selectbox("Jurisdiction", ["GLOBAL","US","EU","UK","SG","MY"])
            regs = get_regulations_by_jurisdiction(jurisdiction)
            
            risks = detect_risks(contract, regs)
            st.markdown("### ⚠️ Missing clauses")
            render_contract_row(contract, risks)
            
            updated_contract_text = contract['text']

            if risks:
                # Collect unique flagged clauses
                unique_clauses = []
                seen = set()
                for r in risks:
                    sentence = r.get('sentence','').strip()
                    if sentence and sentence not in seen:
                        unique_clauses.append(sentence)
                        seen.add(sentence)
                all_flagged_text = "\n\n".join(unique_clauses)
                st.text_area("All Flagged Clauses", value=all_flagged_text, height=300, key="flagged_text_area")

                # ----- Append Amendments (No Timestamps) -----
                if st.button("Append Amendment"):
                    amended_text = updated_contract_text
                    for r in risks:
                        temp_contract = {'text': amended_text, 'name': contract['name']}
                        amended, _ = apply_amendment_append(temp_contract, r)
                        amended_text = amended['text']
                    updated_contract_text = amended_text
                    contract['text'] = amended_text
                    st.success("All amendments appended successfully.")

                # ----- Replace Clauses (No Timestamps) -----
                if st.button("Replace Clause"):
                    amended_text = updated_contract_text
                    for r in risks:
                        temp_contract = {'text': amended_text, 'name': contract['name']}
                        amended, _, replaced = replace_clause_with_amendment(temp_contract, r)
                        amended_text = amended['text']
                    updated_contract_text = amended_text
                    contract['text'] = amended_text
                    st.success("All clauses replaced/appended successfully.")

            # ----- Clean Preview (Remove any existing timestamps) -----
            import re
            clean_preview_text = re.sub(r'\[AMENDMENT - .*?\]\s*', '', updated_contract_text)

            st.markdown("### 👀 Preview Updated Contract (Clean)")
            st.text_area(
                "Updated Contract Text Preview",
                value=clean_preview_text,
                height=400,
                key="preview_updated_contract"
            )

            # ----- Download Updated Contract -----
            st.markdown("### 📥 Download Updated Contract")
            from io import BytesIO
            from reportlab.platypus import SimpleDocTemplate, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet

            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer)
            styles = getSampleStyleSheet()
            story = [Paragraph(clean_preview_text.replace('\n','<br />'), styles["Normal"])]
            doc.build(story)
            pdf_bytes = pdf_buffer.getvalue()

            updated_folder = os.path.join("data", "updated_contracts")
            os.makedirs(updated_folder, exist_ok=True)
            base_name = os.path.splitext(contract['name'])[0]  # remove .pdf
            updated_pdf_path = os.path.join(updated_folder, f"{base_name}_updated.pdf")
            with open(updated_pdf_path, "wb") as f:
                f.write(pdf_bytes)

            st.download_button(
                label="Download Updated Contract PDF",
                data=pdf_bytes,
                file_name=f"{base_name}_updated.pdf",
                mime="application/pdf"
            )

        else:
            st.info("Please select a contract from the dropdown to review and detect risks.")

# ---------------- PAGE: Legal Chatbot ---------------- 
elif page == "Legal Chatbot":
    st.title("🤖 Legal Chatbot")

    # Initialize session states
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    if 'contract_text' not in st.session_state:
        st.session_state.contract_text = ""
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'embedder' not in st.session_state:
        st.session_state.embedder = None

    # Load contracts
    existing_contracts = read_contracts_from_pdfs() or []
    if not existing_contracts:
        st.warning("No contracts found for chat.")
        st.stop()

    # First option should always be default
    contract_names = ["Select contract"] + [c['name'] for c in existing_contracts]

    selected_name = st.selectbox("Select a contract to chat with", contract_names)

    # ------------------------------------------
    # ⛔ STOP everything if "Select contract" chosen
    # ------------------------------------------
    if selected_name == "Select contract":
        st.info("Please select a contract to start the Legal Chatbot.")
        st.stop()

    # Continue normally once a contract is selected
    selected_contract = next(c for c in existing_contracts if c['name'] == selected_name)

    # If new contract selected, prepare embeddings
    if st.session_state.contract_text != selected_contract['text']:
        st.session_state.contract_text = selected_contract['text']
        chunks = split_contract_text(selected_contract['text'])
        st.session_state.chunks = chunks

        if HuggingFaceEmbeddings is not None:
            embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            st.session_state.embedder = embedder

            with st.spinner("Computing embeddings..."):
                vectors = embedder.embed_documents(chunks)

            import faiss, numpy as np
            vectors_np = np.array(vectors).astype('float32')
            dimension = vectors_np.shape[1]

            index = faiss.IndexFlatL2(dimension)
            index.add(vectors_np)

            st.session_state.faiss_index = {'index': index, 'vectors': vectors_np}
            st.success(f"Contract '{selected_name}' indexed for chat!")
        else:
            st.error("HuggingFaceEmbeddings not available. Install langchain-community and sentence-transformers.")

    # Chat input
    user_query = st.chat_input("Ask your legal question here...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        context_text = ""
        if st.session_state.faiss_index is not None and st.session_state.embedder is not None:
            q_vec = st.session_state.embedder.embed_query(user_query)
            import numpy as np

            q_np = np.array([q_vec]).astype('float32')
            index = st.session_state.faiss_index['index']
            D, I = index.search(q_np, 5)

            retrieved_chunks = [
                st.session_state.chunks[idx] for idx in I[0] if idx < len(st.session_state.chunks)
            ]
            context_text = "\n\n".join(retrieved_chunks)

        # Generate response
        if ChatGroq is None:
            bot_answer = f"Based on '{selected_name}':\n\n{context_text}"
        else:
            try:
                llm = ChatGroq(model="llama-3.3-70b-versatile")
                prompt = f"""
                You are a legal assistant. Use the contract text below to answer the user's question.

                Contract Context:
                {context_text}

                Question:
                {user_query}

                If the answer is not in the contract, say "The contract does not contain information about this."
                """
                response = llm.invoke(prompt)
                bot_answer = getattr(response, 'content', str(response))
            except Exception as e:
                bot_answer = f"LLM invocation failed: {e}"

        st.session_state.messages.append({"role": "assistant", "content": bot_answer})

    # Display chat
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='msg-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='msg-bot'>{msg['content']}</div>", unsafe_allow_html=True)

# ---------------- PAGE: Notifications ----------------
elif page == "Notifications":
    st.title("📣 Notifications & Alerts")

    # Get all updated contracts from 'data/updated_contracts'
    updated_folder = os.path.join("data", "updated_contracts")
    if not os.path.exists(updated_folder):
        st.error("No updated contracts found. Please update contracts first.")
    else:
        updated_contract_files = [f for f in os.listdir(updated_folder) if f.lower().endswith(".pdf")]
        names = ["Select updated contract"] + updated_contract_files
        sel = st.selectbox("Select updated contract to notify", names)

        if sel != "Select updated contract":
            updated_contract_path = os.path.join(updated_folder, sel)
            st.write(f"Selected contract: {sel}")

            emails = st.text_area("Recipient emails (comma separated)")

            if st.button("Send Notifications"):
                if not emails.strip():
                    st.error("Enter at least one email.")
                else:
                    recipients = [e.strip() for e in emails.split(',') if e.strip()]
                    if not os.path.exists(updated_contract_path):
                        st.error(f"Updated contract PDF not found: {updated_contract_path}")
                    else:
                        try:
                            # Import from notifier.py
                            from notifier import notify_on_risks_simple
                            
                            message = "Your contract has been updated successfully.\nPlease find the updated contract attached."
                            subject = f"Updated Contract - {sel}"

                            # Send email with attachment
                            notify_on_risks_simple(recipients, subject, message, updated_contract_path)
                            st.success(f"Notifications sent to: {', '.join(recipients)}")
                        except Exception as e:
                            st.error(f"Failed to send notifications: {str(e)}")
        else:
            st.info("Please select an updated contract from the dropdown.")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("<div class='small-note'>Smart compliance, smarter decisions. 🔒✨</div>", unsafe_allow_html=True)