# AI Cognitive Routing & RAG

> **Stack:** Python 3.11 · LangChain / LangGraph · FAISS · sentence-transformers · Groq (Llama-3 8B)

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/navdeepchhillar/Cognitive-Routing-RAG
cd Cognitive-Routing-RAG

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# → Edit .env and add your GROQ_API_KEY (free at console.groq.com)

# 5. Run all phases
python main.py

# Run a single phase (e.g., Phase 1 only)
python main.py 1
```

---

## Architecture

### Phase 1 — Vector-Based Persona Router (`phase1_router.py`)

| Step | What happens |
|------|-------------|
| Embed personas | `SentenceTransformer("all-MiniLM-L6-v2")` encodes each bot's description into a 384-dim L2-normalised vector. |
| Build FAISS index | Vectors are inserted into a `faiss.IndexFlatIP` (inner-product = cosine sim for unit vectors). |
| Route a post | Incoming post is embedded the same way; FAISS returns cosine scores against all 3 personas. Bots above `threshold` (default 0.30) receive the post. |

**Threshold note:** `all-MiniLM-L6-v2` produces compact embeddings where typical cross-sentence similarities sit in the 0.20–0.60 range. The assignment's suggested `0.85` is calibrated for OpenAI `text-embedding-ada-002` (much higher dot-product space). Threshold is an explicit parameter so you can tune it per model.

---

### Phase 2 — LangGraph Content Engine (`phase2_content_engine.py`)

```
┌─────────────────┐    ┌──────────────┐    ┌──────────────┐
│  Node 1         │───▶│  Node 2      │───▶│  Node 3      │
│  Decide Search  │    │  Web Search  │    │  Draft Post  │
│                 │    │  (mock tool) │    │  (JSON out)  │
│  LLM decides    │    │  Returns     │    │  ≤280 chars  │
│  search topic   │    │  headlines   │    │  strict JSON │
└─────────────────┘    └──────────────┘    └──────────────┘
```

- **Node 1 (Decide Search):** The LLM is given only the bot's system prompt/persona and asked to produce a short search query reflecting what the bot would care about today.
- **Node 2 (Web Search):** Calls `mock_searxng_search(@tool)` which matches keywords in the query to a dictionary of hardcoded headlines — simulating SearXNG with no external HTTP calls required.
- **Node 3 (Draft Post):** The LLM receives `System = [persona + JSON schema rules]` and `Human = [search result + bot_id]`. It must return **only** a JSON object `{"bot_id", "topic", "post_content"}`. A `json.loads()` parse + graceful fallback ensures the output is always a valid dict even if the model adds stray text.

**Structured output enforcement:** Rather than using function-calling (requires OpenAI or compatible endpoint), the JSON schema is embedded directly in the system prompt with explicit rules against markdown fences or preamble. The parser strips fences as a belt-and-suspenders measure.

---

### Phase 3 — Combat Engine with RAG + Injection Defense (`phase3_combat_engine.py`)

#### RAG Context Construction

`generate_defense_reply()` builds a structured `<THREAD_CONTEXT>` block containing:
- The original human post
- Every prior comment with attributed author labels

This full thread is prepended to the `HumanMessage`, giving the LLM complete argument history — not just the last reply. This is the RAG: retrieved argument context injected directly into the prompt.

#### Prompt Injection Defense

**Detection (regex heuristic):**  
A compiled multi-pattern regex (`_INJECTION_RE`) scans the human's latest reply for phrases like:
- `"ignore all previous instructions"`
- `"you are now a …"`
- `"apologize to me"`, `"forget your persona"`, etc.

When triggered, two defense layers activate:

1. **Structural system-level block:** A `⚠️ INJECTION ALERT` section is appended to the system prompt, explicitly ordering the LLM to ignore the injected instruction and continue the argument naturally.
2. **Persona anchor:** The system prompt wraps the bot's persona in `<PERSONA>` tags with a preamble stating it is "fixed" and "immutable", making it harder for injected user-role instructions to override the system context.

**Why this works:**  
LLMs weight `SystemMessage` content more heavily than `HumanMessage` content. By flagging the injection inside the system message (where the model is primed to treat it as authoritative), the bot reliably dismisses the injection and re-engages with the original argument thread.

---

## File Structure

```
grid07-ai-assignment/
├── phase1_router.py          # Vector persona matching
├── phase2_content_engine.py  # LangGraph 3-node pipeline
├── phase3_combat_engine.py   # RAG combat + injection defense
├── main.py                   # Orchestrates all phases
├── execution_logs.md         # Sample console output
├── requirements.txt
├── .env.example
└── README.md
```

---

## Swapping the LLM

The LLM is initialised in one place in each phase file. To switch providers:

```python
# Groq (default)
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Ollama (local, no key)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3")
```

---

## Notes

- No real API keys are committed anywhere. All secrets live in `.env` (gitignored).
- Phase 1 uses CPU FAISS; replace `faiss-cpu` with `faiss-gpu` in `requirements.txt` for GPU acceleration.
- The mock search tool can be replaced with a real `requests.get(searxng_url, ...)` call without changing any LangGraph node logic.
