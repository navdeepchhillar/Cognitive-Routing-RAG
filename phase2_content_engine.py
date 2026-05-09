"""
Phase 2: The Autonomous Content Engine (LangGraph)
---------------------------------------------------
A three-node LangGraph state machine that:
  1. Decides what to search based on the bot's persona.
  2. Executes a mock SearXNG search for real-world context.
  3. Drafts a 280-char opinionated post and returns strict JSON output.
"""

import json
import os
from typing import TypedDict, Annotated

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq          # swap for ChatOpenAI if preferred
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# LLM setup  (reads GROQ_API_KEY from env; swap for OpenAI or Ollama easily)
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ---------------------------------------------------------------------------
# Mock SearXNG search tool
# ---------------------------------------------------------------------------
MOCK_NEWS_DB = {
    "crypto":       "Bitcoin hits new all-time high amid regulatory ETF approvals; Ethereum layer-2 volume surges 300%.",
    "bitcoin":      "Bitcoin crosses $100k as BlackRock files for spot BTC ETF expansion.",
    "ai":           "OpenAI releases GPT-5 with autonomous agent capabilities, sparking mass layoff fears in tech.",
    "elon":         "Elon Musk announces Grok-3 integration with Tesla Autopilot; SpaceX Starship completes orbital flight.",
    "space":        "NASA's Artemis III crew announced; private moon economy projected at $500B by 2040.",
    "privacy":      "EU fines Meta €1.2B for GDPR violations; Apple expands end-to-end encryption across iCloud.",
    "surveillance": "New report reveals ISPs selling browsing data to data brokers without user consent.",
    "capitalism":   "Record CEO-to-worker pay ratio hits 350:1; union membership rises for first time in 30 years.",
    "market":       "S&P 500 rallies 2% on Fed pivot signals; options market pricing in 40bps cut by Q3.",
    "interest":     "Fed holds rates at 5.25%; bond yield curve finally un-inverts after 18 months.",
    "trading":      "Quant hedge funds outperform market by 18% using transformer-based momentum models.",
    "roi":          "Private equity returns shrink to 8% average as leverage costs bite into portfolio valuations.",
}

@tool
def mock_searxng_search(query: str) -> str:
    """Simulates a SearXNG web search and returns relevant news headlines."""
    query_lower = query.lower()
    for keyword, headline in MOCK_NEWS_DB.items():
        if keyword in query_lower:
            return f"[SEARCH RESULT] {headline}"
    return "[SEARCH RESULT] No specific results found. General tech and finance markets remain volatile."


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    bot_id: str
    persona: str
    search_query: str
    search_result: str
    # LangGraph managed message history
    messages: Annotated[list, add_messages]
    final_post: dict   # {"bot_id": ..., "topic": ..., "post_content": ...}


# ---------------------------------------------------------------------------
# Node 1 — Decide Search
# ---------------------------------------------------------------------------
def node_decide_search(state: AgentState) -> AgentState:
    """LLM decides what to search based on the bot's persona."""
    system = SystemMessage(content=(
        "You are the following social media bot persona:\n"
        f"{state['persona']}\n\n"
        "Your task: Decide ONE topic you want to post about today and output "
        "ONLY a short search query (3-6 words, no punctuation)."
    ))
    human = HumanMessage(content="What do you want to search for today? Respond with only the search query.")
    response: AIMessage = llm.invoke([system, human])
    query = response.content.strip().strip('"').strip("'")
    print(f"  [Node 1 — Decide Search]  query: \"{query}\"")
    return {**state, "search_query": query, "messages": [system, human, response]}


# ---------------------------------------------------------------------------
# Node 2 — Web Search
# ---------------------------------------------------------------------------
def node_web_search(state: AgentState) -> AgentState:
    """Executes the mock search tool."""
    result = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"  [Node 2 — Web Search]     result: {result}")
    return {**state, "search_result": result}


# ---------------------------------------------------------------------------
# Node 3 — Draft Post
# ---------------------------------------------------------------------------
_JSON_SCHEMA = '{"bot_id": "<string>", "topic": "<string>", "post_content": "<string max 280 chars>"}'

def node_draft_post(state: AgentState) -> AgentState:
    """LLM drafts a 280-char opinionated post and returns strict JSON."""
    system = SystemMessage(content=(
        "You are the following social media bot persona:\n"
        f"{state['persona']}\n\n"
        "Rules:\n"
        "1. Stay fully in character — opinionated, unapologetic, authentic to your persona.\n"
        "2. Use the provided search result as context.\n"
        "3. Write a post that is AT MOST 280 characters.\n"
        "4. Respond ONLY with a valid JSON object matching this schema exactly:\n"
        f"   {_JSON_SCHEMA}\n"
        "5. Do NOT include markdown fences, comments, or any text outside the JSON."
    ))
    human = HumanMessage(content=(
        f"Search result context: {state['search_result']}\n\n"
        f"Bot ID: {state['bot_id']}\n"
        "Write your post now."
    ))
    response: AIMessage = llm.invoke([system, human])
    raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        post_obj = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: wrap raw text in valid structure
        post_obj = {
            "bot_id": state["bot_id"],
            "topic": state["search_query"],
            "post_content": raw[:280],
        }

    # Enforce 280-char limit on post_content
    post_obj["post_content"] = post_obj.get("post_content", "")[:280]
    print(f"  [Node 3 — Draft Post]     output: {json.dumps(post_obj, indent=2)}")
    return {**state, "final_post": post_obj}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_content_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("decide_search", node_decide_search)
    g.add_node("web_search",    node_web_search)
    g.add_node("draft_post",    node_draft_post)

    g.set_entry_point("decide_search")
    g.add_edge("decide_search", "web_search")
    g.add_edge("web_search",    "draft_post")
    g.add_edge("draft_post",    END)
    return g.compile()


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------
def generate_bot_post(bot_id: str, persona: str) -> dict:
    """Run the full LangGraph pipeline for a single bot and return the JSON post."""
    graph = build_content_graph()
    initial_state: AgentState = {
        "bot_id": bot_id,
        "persona": persona,
        "search_query": "",
        "search_result": "",
        "messages": [],
        "final_post": {},
    }
    final_state = graph.invoke(initial_state)
    return final_state["final_post"]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from phase1_router import BOT_PERSONAS

    print("=" * 65)
    print("PHASE 2 — AUTONOMOUS CONTENT ENGINE")
    print("=" * 65)

    for bot_id, info in BOT_PERSONAS.items():
        print(f"\n>>> Running pipeline for {bot_id} ({info['name']}) …")
        result = generate_bot_post(bot_id, info["description"])
        print(f"\n  FINAL JSON OUTPUT:\n  {json.dumps(result, indent=4)}")
        print("-" * 65)
