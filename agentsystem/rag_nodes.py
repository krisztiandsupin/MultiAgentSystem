from langchain_ollama import ChatOllama
from .vector_store import VectorDB

db = VectorDB(model="nomic-embed-text")
llm = ChatOllama(model="llama3")

def rag_retriever(state):
    docs = db.search(state["task"])
    state["context"] = "\n---\n".join([d.text for d,_ in docs])
    return state

def rag_writer(state):
    prompt=f"Context:\n{state.get('context')}\n\nTask:\n{state['task']}\n\nAnswer:"
    state["draft"]=llm.invoke(prompt).content
    return state