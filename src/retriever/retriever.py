import logging
import asyncio
from typing import List
from src.app.config import logging

class Retriever:
    def __init__(self, vector_store, k: int = 6):
        self.vector_store = vector_store
        self.k = k

    def get_retriever(self):
        # ensure the actual DB object is available
        return self.vector_store.as_retriever(search_kwargs={"k": self.k})

    def __call__(self, question: str) -> str:
        """
        Make Retriever callable for Runnable chains.
        Robustly calls whichever retrieval API is available on the underlying retriever,
        handling different LangChain versions (sync/async, public/private API names).
        Returns a single formatted context string for the prompt.
        """
        """Make retriever callable by Runnable chain; extract the text query if input is a dict."""
    # 🔹 Some LangChain Runnables pass a full dict — extract the question text.
        if isinstance(question, dict):
            question = question.get("question") or question.get("input") or str(question)
        if not isinstance(question, str):
            question = str(question)

        retriever = self.get_retriever()
        docs: List = []

        # 1) Prefer public synchronous method if present
        if hasattr(retriever, "get_relevant_documents"):
            try:
                docs = retriever.get_relevant_documents(question)
            except TypeError as e:
                # some versions accept run_manager kw-only
                logging.debug("get_relevant_documents raised TypeError, retrying with run_manager=None: %s", e)
                try:
                    docs = retriever.get_relevant_documents(question, run_manager=None)
                except Exception as e2:
                    logging.exception("get_relevant_documents failed even with run_manager=None: %s", e2)
                    docs = []
            except Exception as e:
                logging.exception("get_relevant_documents failed: %s", e)
                docs = []

        # 2) Fallback to private sync API used in some versions
        elif hasattr(retriever, "_get_relevant_documents"):
            try:
                docs = retriever._get_relevant_documents(question)
            except TypeError as e:
                logging.debug("_get_relevant_documents requires run_manager; calling with run_manager=None: %s", e)
                try:
                    docs = retriever._get_relevant_documents(question, run_manager=None)
                except Exception as e2:
                    logging.exception("_get_relevant_documents failed even with run_manager=None: %s", e2)
                    docs = []
            except Exception as e:
                logging.exception("_get_relevant_documents failed: %s", e)
                docs = []

        # 3) Fallback to async method if present (aget_relevant_documents or similar)
        else:
            a_fn = getattr(retriever, "aget_relevant_documents", None) or getattr(retriever, "get_relevant_documents_async", None)
            if a_fn:
                try:
                    # run the coroutine to completion (sync context)
                    docs = asyncio.get_event_loop().run_until_complete(a_fn(question))
                except RuntimeError:
                    # no running loop -> create one
                    loop = asyncio.new_event_loop()
                    try:
                        docs = loop.run_until_complete(a_fn(question))
                    finally:
                        loop.close()
                except Exception as e:
                    logging.exception("Async retrieval failed: %s", e)
                    docs = []
            else:
                # last resort: try calling the retriever if it's itself callable
                try:
                    docs = retriever(question)
                except Exception as e:
                    logging.exception("Retriever call fallback failed: %s", e)
                    docs = []

        # Build context string from retrieved docs (safe formatting)
        snippets = []
        for d in docs or []:
            try:
                meta = getattr(d, "metadata", {}) or {}
                src = meta.get("source") or meta.get("title") or "unknown"
                chunk_idx = meta.get("chunk_index", "n/a")
                snippet = (getattr(d, "page_content", "") or "")[:500].replace("\n", " ")
                snippets.append(f"[{src}] [chunk:{chunk_idx}] {snippet}")
            except Exception as e:
                logging.debug("Skipping a document while building context: %s", e)

        return "\n\n".join(snippets) if snippets else "No context available."
