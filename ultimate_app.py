# ==========================
# 🚀 ULTIMATE RAG SYSTEM (LangGraph)
# ==========================

import os
import streamlit as st
import asyncio
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

# --------------------------
# 🔑 CONFIG
# --------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --------------------------
# 🧠 MODELS
# --------------------------
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from pinecone_text.sparse import BM25Encoder

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25 = BM25Encoder().default()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --------------------------
# 🧠 PINECONE
# --------------------------
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX = "ultimate-rag"

if INDEX not in pc.list_indexes().names():
    pc.create_index(name=INDEX, dimension=384, metric="dotproduct",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))

index = pc.Index(INDEX)

# --------------------------
# 🔹 RETRIEVER
# --------------------------
from langchain_community.retrievers import PineconeHybridSearchRetriever

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25,
    index=index
)

# --------------------------
# 🧠 MEMORY
# --------------------------
def store_memory(user_id, query, response):
    retriever.add_texts(
        [f"{query} -> {response}"],
        metadatas=[{"user_id": user_id, "type": "memory"}],
        ids=[str(uuid4())]
    )

def get_memory(user_id, query):
    docs = retriever.invoke(query)
    return "\n".join([
        d.page_content for d in docs
        if d.metadata.get("user_id") == user_id
    ][:3])

# --------------------------
# 🔥 TOOLS
# --------------------------
async def vector_tool(query):
    return retriever.invoke(query)

async def web_tool(query):
    import httpx
    from bs4 import BeautifulSoup

    async with httpx.AsyncClient() as client:
        res = await client.get(f"https://html.duckduckgo.com/html/?q={query}")
        soup = BeautifulSoup(res.text, "html.parser")
        return [type("Doc", (), {"page_content": soup.get_text()[:1000]})]

TOOLS = {
    "vector": vector_tool,
    "web": web_tool,
}

# --------------------------
# 🧠 LANGGRAPH STATE
# --------------------------
from typing import TypedDict, List

class State(TypedDict):
    query: str
    user_id: str
    docs: List
    answer: str
    tool: str

# --------------------------
# 🤖 AGENT NODES
# --------------------------

async def planner(state):
    query = state["query"]

    response = await llm.ainvoke(
        f"""
        You are a planner agent name rio once your task is done call Ryan with appropriate tool
        Decide which tool to use.

        Available tools:
        - vector : for knowledge base, documents, memory
        - web : for current web information

        Query: {query}

        Return only:
        vector
        OR
        web
        """
    )

    state["tool"] = response.content.strip().lower()
    return state


async def tool_executor(state):
    tool = state["tool"]
    query = state["query"]

    if tool in TOOLS:
        docs = await TOOLS[tool](query)
    else:
        docs = []

    state["docs"] = docs
    return state


async def reranker_node(state):
    docs = state["docs"]

    if not docs:
        return state

    query = state["query"]

    pairs = [
        (query, d.page_content)
        for d in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    state["docs"] = [
        d for d, _ in ranked[:5]
    ]

    return state


async def generator(state):
    query = state["query"]
    docs = state["docs"]
    user_id = state["user_id"]

    context = "\n".join(
        [d.page_content for d in docs]
    )

    memory = get_memory(user_id, query)

    prompt = f"""
    You are an helful assistant Ryan who will give answer based on the context in a concise way after review if details are not required.
    Memory:
    {memory}

    Context:
    {context}

    Question:
    {query}

    Answer in a detailed and helpful way.
    """

    response = await llm.ainvoke(prompt)

    state["answer"] = response.content

    store_memory(
        user_id,
        query,
        response.content
    )

    return state

# --------------------------
# 🔗 BUILD GRAPH
# --------------------------
from langgraph.graph import StateGraph

graph = StateGraph(State)

graph.add_node("planner", planner)
graph.add_node("tool", tool_executor)
graph.add_node("rerank", reranker_node)
graph.add_node("generate", generator)

graph.set_entry_point("planner")

graph.add_edge("planner", "tool")
graph.add_edge("tool", "rerank")
graph.add_edge("rerank", "generate")

app = graph.compile()

# --------------------------
# 🎨 STREAMLIT UI
# --------------------------
st.title("🔥 Ultimate AI Assistant")

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid4())

query = st.text_input("Ask anything:")

async def run_graph(query, user_id): 
    return await app.ainvoke({
        "query": query,
        "user_id": user_id,
        "docs": [],
        "answer": "",
        "tool": ""
    })

if st.button("Ask") and query:

    with st.spinner("Thinking..."):
        result = asyncio.run(
            run_graph(
                query,
                st.session_state.user_id
            )
        )

    st.write(result["answer"])