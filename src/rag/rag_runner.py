from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from typing import List, Optional
from dotenv import load_dotenv
#from src.ingest.indexer import VectorStore
from src.retriever.vector_store import VectorStore
from src.retriever.retriever import Retriever
from src.app.config import logging, PERSIST_DIR, CHAT_MODEL, EMBEDDING_MODEL, PROMPT
from langchain_core.callbacks import BaseCallbackHandler



load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class DebugLLMMessagesCallback(BaseCallbackHandler):
    """Logs final messages sent to the LLM."""
    def on_llm_start(self, serialized, prompts, **kwargs):
        logging.info("=== FINAL PROMPTS TO LLM (start) ===")
        for i, p in enumerate(prompts):
            logging.info("Prompt %d:\n%s", i, p.replace("\n", " ")[:2000])
        logging.info("=== FINAL PROMPTS TO LLM (end) ===")


class RAGRunner:
    def __init__(self, k: int = 6):
        self.vector_store = VectorStore(
            persist_dir=PERSIST_DIR,
            embedding_model=EMBEDDING_MODEL
        )
        #self.retriever = Retriever(self.vector_store, k=k)  # callable retriever
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, verbose=True, callbacks=[DebugLLMMessagesCallback()])
        self.rag_chain = None

    def init_persisted_db(self):
        loaded_vector_store = self.vector_store.load_vector_db()
        if loaded_vector_store:
            self.vector_store = loaded_vector_store
            #self.retriever = Retriever(self.vector_store, k=self.retriever.k)
            self.retriever = loaded_vector_store.as_retriever()
            logging.info("Loaded persisted vector DB and recreated retriever.")
        else:
            logging.info("No persisted DB found; continuing with current retriever.")
        return self.retriever
    

    def _build_history_aware_components(self):
        """
        Build history-aware retriever + retrieval->qa chain, similar to your notebook.
        Returns rag_chain_ready that accepts {'input': question, 'chat_history': chat_history_msgs}
        """

        loaded_vector_store = self.vector_store.load_vector_db()
        retriever = loaded_vector_store.as_retriever()

        # 1) contextualizer prompt: reformulates follow-ups to standalone question
        contextualize_q_system_prompt = (
            """
            Given a chat history and the latest user question, which might reference context in the chat history,
            formulate a standalone question that includes any specific facts or entity mentions from the history
            (e.g. issue IDs, instrument names, codes). Keep entity names and IDs verbatim. Do NOT answer the question,
            just return the reformulated standalone question. If no reformulation is needed, return the original question.
            """
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 2) create the history-aware retriever that first runs the above LLM reformulation
        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)

        # 3) QA prompt (you probably already have PROMPT in config; ensure it uses MessagesPlaceholder('chat_history') if desired)
        # Example: PROMPT should be ChatPromptTemplate.from_messages([... , MessagesPlaceholder("chat_history"), ("human","{input}") ...])
        system_prompt = PROMPT  # reuse your existing prompt that expects chat_history and input

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("system", "IMPORTANT: Use the conversation history above when answering."),
                ("human", "{question}"),
            ]
        )

        # 4) create the chain that stuffs retrieved docs into LLM (same as notebook)
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # 5) combine into retrieval chain
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        return self.rag_chain


    def _build_runnable_chain(self):
        """Compose the runnable chain once."""
        # ensure persisted DB is ready
        retriever = self.init_persisted_db()

        # The retriever is callable (__call__), so RunnableParallel accepts it
        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | PROMPT
            | self.llm
            | StrOutputParser()
        )

        logging.info("Runnable RAG chain built successfully.")
        return self.rag_chain
    


    def answer(self, question: str, chat_history: Optional[List] = None, debug: bool = False):
        chat_history = chat_history or []
        logging.info("RAG.answer() called question=%s len(chat_history)=%d", question[:120], len(chat_history))

        loaded_vector_store = self.vector_store.load_vector_db()
        retriever = loaded_vector_store.as_retriever()

        if not hasattr(self, "_history_rag_chain") or self._history_rag_chain is None:
            self._history_rag_chain = self._build_history_aware_components()

        # Convert incoming chat_history into Message objects
        msgs = []
        if chat_history:
            for turn in chat_history:
                if isinstance(turn, (list, tuple)):
                    user_msg = turn[0] if len(turn) > 0 else None
                    assistant_msg = turn[1] if len(turn) > 1 else None
                    if user_msg:
                        msgs.append(HumanMessage(content=str(user_msg)))
                    if assistant_msg:
                        msgs.append(AIMessage(content=str(assistant_msg)))
                elif isinstance(turn, dict) and "role" in turn and "content" in turn:
                    if turn["role"] == "user":
                        msgs.append(HumanMessage(content=str(turn["content"])))
                    elif turn["role"] in ("assistant", "system"):
                        msgs.append(AIMessage(content=str(turn["content"])))
                elif hasattr(turn, "content"):
                    msgs.append(turn)
                else:
                    msgs.append(HumanMessage(content=str(turn)))

        # debug logging of converted messages
        logging.info("Converted chat_history -> %d Message objects", len(msgs))
        for i, m in enumerate(msgs):
            role = "user" if isinstance(m, HumanMessage) else "assistant" if isinstance(m, AIMessage) else "unknown"
            logging.info(" msg[%d] role=%s content=%s", i, role, (m.content or "")[:300])

        # --- Build combined_context (retrieved docs + conversation history) ---
        docs = retriever.get_relevant_documents(question) if hasattr(retriever, "get_relevant_documents") else []
        docs_text = "\n\n".join([d.page_content for d in docs if getattr(d, "page_content", None)])

        hist_text_lines = []
        for m in msgs:
            if isinstance(m, HumanMessage):
                hist_text_lines.append(f"User: {m.content}")
            elif isinstance(m, AIMessage):
                hist_text_lines.append(f"Assistant: {m.content}")
        hist_text = "\n".join(hist_text_lines)

        if docs_text and hist_text:
            combined_context = docs_text + "\n\nConversation history:\n" + hist_text
        elif docs_text:
            combined_context = docs_text
        else:
            combined_context = "Conversation history:\n" + hist_text if hist_text else ""

        logging.info("Combined context length=%d (docs_text=%d, hist_text=%d)",
                    len(combined_context), len(docs_text), len(hist_text))

        # If we have conversation history, do a direct LLM call with PROMPT filled by combined_context.
        # This bypasses the internal QA runnable which was discarding the chat history.
        answer_text = None
        used_direct_llm = False
        if len(msgs) > 0:
            try:
                # Prepare system prompt by injecting combined_context into PROMPT
                system_prompt_text = PROMPT.replace("{context}", combined_context) if isinstance(PROMPT, str) else str(PROMPT).replace("{context}", combined_context)

                # Build final message list: system then human question.
                final_messages = [SystemMessage(content=system_prompt_text), HumanMessage(content=question)]

                # Call the chat LLM directly - this should always send the messages we constructed.
                logging.info("Calling LLM directly with system+human messages (direct path).")
                llm_resp = self.llm(final_messages)  # ChatOpenAI accepts a list of Message objects

                # Try multiple ways to extract text (be defensive across langchain versions)
                if isinstance(llm_resp, list):
                    # Sometimes returns [AIMessage(...)]
                    first = llm_resp[0]
                    answer_text = getattr(first, "content", str(first))
                elif hasattr(llm_resp, "generations"):
                    gens = llm_resp.generations
                    # gens may be list of lists or list of Generation objects
                    if isinstance(gens, list) and len(gens) > 0:
                        first = gens[0]
                        if isinstance(first, list) and len(first) > 0:
                            answer_text = getattr(first[0], "text", str(first[0]))
                        else:
                            answer_text = getattr(first, "text", str(first))
                elif hasattr(llm_resp, "content"):
                    answer_text = llm_resp.content
                else:
                    answer_text = str(llm_resp)

                used_direct_llm = True
                logging.info("Direct LLM returned %d chars", len(answer_text or ""))
            except Exception as e:
                logging.exception("Direct LLM call failed, falling back to chain: %s", e)
                used_direct_llm = False

        # If direct LLM was not used or failed, use the existing chain path (pass combined_context as 'context')
        if not used_direct_llm:
            inputs = {
                "input": question,
                "question": question,
                "context": combined_context,
                "chat_history": msgs
            }
            logging.info("Invoking chain with keys: %s", list(inputs.keys()))
            result = self._history_rag_chain.invoke(inputs)
            if isinstance(result, dict):
                answer_text = result.get("answer") or result.get("output") or str(result)
            else:
                answer_text = str(result)

        # optionally attach debug_history (simple dicts)
        debug_history = None
        if debug:
            debug_history = []
            for m in msgs:
                debug_history.append({
                    "role": "user" if isinstance(m, HumanMessage) else "assistant" if isinstance(m, AIMessage) else "unknown",
                    "content": m.content
                })

        # return sources / answer as before, plus debug_history if requested
        sources = [{"source": (d.metadata or {}).get("source"), "snippet": (d.page_content or "")[:300]} for d in docs]

        out = {"answer": answer_text, "sources": sources, "file_url": "/mnt/data/test.ipynb", "used_direct_llm": used_direct_llm}
        if debug:
            out["debug_history"] = debug_history
        return out
    

    """
    def answer(self, question: str, chat_history: Optional[List] = None, debug: bool = False):
        chat_history = chat_history or []
        logging.info("RAG.answer() called question=%s len(chat_history)=%d", question[:120], len(chat_history))

        loaded_vector_store = self.vector_store.load_vector_db()
        retriever = loaded_vector_store.as_retriever()

        if not hasattr(self, "_history_rag_chain") or self._history_rag_chain is None:
            self._history_rag_chain = self._build_history_aware_components()

        # Convert incoming chat_history into Message objects
        msgs = []
        if chat_history:
            for turn in chat_history:
                if isinstance(turn, (list, tuple)):
                    user_msg = turn[0] if len(turn) > 0 else None
                    assistant_msg = turn[1] if len(turn) > 1 else None
                    if user_msg:
                        msgs.append(HumanMessage(content=str(user_msg)))
                    if assistant_msg:
                        msgs.append(AIMessage(content=str(assistant_msg)))
                elif isinstance(turn, dict) and "role" in turn and "content" in turn:
                    if turn["role"] == "user":
                        msgs.append(HumanMessage(content=str(turn["content"])))
                    elif turn["role"] in ("assistant", "system"):
                        msgs.append(AIMessage(content=str(turn["content"])))
                elif hasattr(turn, "content"):
                    msgs.append(turn)
                else:
                    msgs.append(HumanMessage(content=str(turn)))

        # debug logging of converted messages
        logging.info("Converted chat_history -> %d Message objects", len(msgs))
        for i, m in enumerate(msgs):
            role = "user" if isinstance(m, HumanMessage) else "assistant" if isinstance(m, AIMessage) else "unknown"
            logging.info(" msg[%d] role=%s content=%s", i, role, (m.content or "")[:300])

        # call chain (supply both keys for compatibility)
        inputs = {"input": question, "question": question, "chat_history": msgs}
        logging.info("Invoking chain with keys: %s", list(inputs.keys()))
        result = self._history_rag_chain.invoke(inputs)

        # Build output
        answer_text = None
        if isinstance(result, dict):
            answer_text = result.get("answer") or result.get("output") or str(result)
        else:
            answer_text = str(result)

        # optionally attach debug_history (simple dicts)
        debug_history = None
        if debug:
            debug_history = []
            for m in msgs:
                debug_history.append({
                    "role": "user" if isinstance(m, HumanMessage) else "assistant" if isinstance(m, AIMessage) else "unknown",
                    "content": m.content
                })

        # return sources / answer as before, plus debug_history if requested
        # (you can keep your existing source-generation logic)
        docs = retriever.get_relevant_documents(question) if hasattr(retriever, "get_relevant_documents") else []
        sources = [{"source": (d.metadata or {}).get("title"), "snippet": (d.page_content or "")[:300]} for d in docs]

        out = {"answer": answer_text, "sources": sources, "file_url": "/mnt/data/test.ipynb"}
        if debug:
            out["debug_history"] = debug_history
        return out
"""


    







    

        
    