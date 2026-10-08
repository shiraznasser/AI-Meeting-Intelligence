"""AI Meeting Intelligence - RAG + Chains + Output Parser + Memory (Streamlit)."""
import os, re, json, tempfile
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema
except ImportError:  # langchain 1.x
    from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

MODEL_NAME = os.environ.get("MEETING_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="AI Meeting Intelligence", page_icon="📝", layout="wide")

# ----------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading language model (first time takes a few minutes)...")
def load_llm():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    return tok, model

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_NAME, model_kwargs={"device": "cpu"},
                                 encode_kwargs={"normalize_embeddings": True})

def generate_text(prompt, max_new_tokens=600, creative=False):
    tok, model = load_llm()
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    if creative:
        kwargs.update(do_sample=True, temperature=0.4, top_p=0.9, repetition_penalty=1.1)
    else:
        kwargs.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ------------------------------------------------- output parsers (schemas)
SUMMARY_SCHEMAS = [
    ResponseSchema(name="summary", description="a 3-4 sentence summary of the meeting"),
    ResponseSchema(name="key_points", description="list of 3-6 short strings, the key points discussed"),
]
DECISION_SCHEMAS = [
    ResponseSchema(name="decisions", description="list of strings, each one a decision explicitly agreed in the meeting. Empty list if none"),
]
TASK_SCHEMAS = [
    ResponseSchema(name="action_items", description="list of objects with keys task, owner, deadline. Use 'Not specified' if the owner or deadline is not mentioned. Empty list if none"),
]

EXTRACT_PROMPT = PromptTemplate(
    input_variables=["context", "task", "format_instructions"],
    template=(
        "You are an AI meeting assistant. Use ONLY the meeting transcript excerpts below. "
        "Do not invent information.\n\nTranscript excerpts:\n{context}\n\n"
        "Task: {task}\n\n{format_instructions}\n\nReturn only the JSON block."
    ),
)
QA_PROMPT = PromptTemplate(
    input_variables=["context", "history", "question"],
    template=(
        "You are an AI meeting assistant. Answer the question using ONLY the transcript excerpts. "
        "If the answer is not in the excerpts, say you cannot find it in the meeting.\n\n"
        "Transcript excerpts:\n{context}\n\nConversation so far:\n{history}\n\n"
        "Question: {question}\nAnswer:"
    ),
)

# ------------------------------------------------------------------- RAG
def load_text(uploaded):
    if uploaded.name.lower().endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded.read())
            path = f.name
        return "\n".join(d.page_content for d in PyPDFLoader(path).load())
    return uploaded.read().decode("utf-8", errors="ignore")

def build_index(text, chunk_size):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    docs = splitter.create_documents([text])
    for i, d in enumerate(docs):
        d.metadata["chunk"] = i
    return FAISS.from_documents(docs, load_embeddings()), len(docs)

def retrieve(vs, query, k):
    docs = vs.similarity_search(query, k=k)
    docs = sorted(docs, key=lambda d: d.metadata["chunk"])  # keep chronological order
    return "\n---\n".join(d.page_content for d in docs)

# ----------------------------------------------------------------- chains
def parse_output(parser, raw):
    try:
        return parser.parse(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}

def run_chain(vs, k, retrieval_query, task, schemas, max_new_tokens=500):
    parser = StructuredOutputParser.from_response_schemas(schemas)
    context = retrieve(vs, retrieval_query, k)
    prompt = EXTRACT_PROMPT.format(context=context, task=task,
                                   format_instructions=parser.get_format_instructions())
    return parse_output(parser, generate_text(prompt, max_new_tokens))

def analyze_meeting(vs, k):
    s = run_chain(vs, k, "overall topics discussed in the meeting",
                  "Summarize the meeting and list its key points.", SUMMARY_SCHEMAS)
    d = run_chain(vs, k, "decision agreed we decided",
                  "Extract every decision that was explicitly agreed.", DECISION_SCHEMAS)
    t = run_chain(vs, k, "please will deliver by deadline task responsible",
                  "Extract all action items with the task, the owner and the deadline.", TASK_SCHEMAS, 700)
    items = t.get("action_items", [])
    if isinstance(items, dict):
        items = [items]
    items = [i if isinstance(i, dict) else {"task": str(i), "owner": "Not specified", "deadline": "Not specified"}
             for i in items]
    return {
        "summary": s.get("summary", ""),
        "key_points": s.get("key_points", []),
        "decisions": d.get("decisions", []),
        "action_items": items,
    }

def answer_question(vs, k, question, chat):
    history = "\n".join(f"Q: {q}\nA: {a}" for q, a in chat[-4:]) or "(none)"  # simple buffer memory
    context = retrieve(vs, question, k)
    return generate_text(QA_PROMPT.format(context=context, history=history, question=question),
                         400, creative=True)

# -------------------------------------------------------------------- UI
st.title("📝 AI Meeting Intelligence")
st.caption("RAG + Chains + Output Parser + Memory — turn a meeting transcript into a structured summary, decisions and tasks.")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks retrieved per chain (k)", 2, 10, 6)
    chunk_size = st.select_slider("Chunk size (characters)", [400, 600, 800, 1000], value=800)
    st.markdown(f"**LLM:** `{MODEL_NAME}`  \n**Embeddings:** `{EMBED_NAME}`")

source = st.radio("Transcript source", ["Use sample", "Upload file (.txt / .pdf)", "Paste text"], horizontal=True)
text = ""
if source == "Use sample":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_meeting.txt")
    text = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
elif source.startswith("Upload"):
    up = st.file_uploader("Upload transcript", type=["txt", "pdf"])
    if up:
        text = load_text(up)
else:
    text = st.text_area("Paste transcript here", height=250)

if text:
    with st.expander("Transcript preview"):
        st.text(text[:3000] + ("..." if len(text) > 3000 else ""))

if st.button("Analyze meeting", type="primary", disabled=not text.strip()):
    with st.spinner("Indexing transcript..."):
        st.session_state.vs, n = build_index(text, chunk_size)
    st.session_state.chat = []
    with st.spinner(f"Running 3 chains over {n} chunks (summary → decisions → tasks)..."):
        st.session_state.result = analyze_meeting(st.session_state.vs, k)

if "result" in st.session_state:
    r = st.session_state.result
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Decisions & Tasks", "Ask the meeting", "JSON"])
    with tab1:
        st.subheader("Summary")
        st.write(r["summary"] or "—")
        st.subheader("Key points")
        for p in r["key_points"]:
            st.markdown(f"- {p}")
    with tab2:
        st.subheader("Decisions")
        for dcs in r["decisions"] or ["No decisions found."]:
            st.markdown(f"- ✅ {dcs}")
        st.subheader("Action items")
        if r["action_items"]:
            st.table(r["action_items"])
        else:
            st.write("No action items found.")
    with tab3:
        for q, a in st.session_state.chat:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
        q = st.chat_input("Ask something about the meeting (follow-ups work too)")
        if q:
            with st.spinner("Thinking..."):
                a = answer_question(st.session_state.vs, k, q, st.session_state.chat)
            st.session_state.chat.append((q, a))
            st.rerun()
    with tab4:
        js = json.dumps(r, indent=2, ensure_ascii=False)
        st.json(r)
        st.download_button("Download JSON", js, "meeting_analysis.json", "application/json")
