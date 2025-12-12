from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from typing import List, Iterable, Optional
from src.app.config import CHUNK_SIZE, CHUNK_OVERLAP, logging

# Optional progress bar if available
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x: x  # noop iterator

"""
class Chunker:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP

    def chunk_documents(self, docs: List[Document]):
        
        #Token-aware chunking using LangChain TokenTextSplitter.
        #Returns list[Document] with chunked page_content and preserved metadata.
        

        splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunked = []

        for doc in docs:
            text = doc.page_content
            chunks = splitter.split_text(text)
            for i, c in enumerate(chunks):
                metadata = dict(doc.metadata or {})
                # include chunk index to help with provenance
                metadata["chunk_index"] = i
                chunked.append(Document(page_content=c, metadata=metadata))
        return chunked
"""

class Chunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Chunker for LangChain Documents.

        Args:
            chunk_size: number of tokens (if using token splitter) or characters (if using char splitter).
            chunk_overlap: overlap between chunks (tokens or chars depending on splitter).
            use_token_splitter: prefer token-aware splitter; if missing, fall back to RecursiveCharacterTextSplitter.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _make_splitter(self):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separators=["\n\n", "\n", " ", ""])
            # test a tiny split to ensure token backend available
            _ = splitter.split_text("test")
            return splitter
        except Exception as e:
            logging.warning("TokenTextSplitter unavailable or failed: %s", e)

    
    def chunk_documents(self, docs: Iterable[Document]) -> List[Document]:
        """
        Token-aware chunking using LangChain splitters.
        Returns a list[Document] with chunked page_content and preserved metadata.
        """
        splitter = self._make_splitter()
        chunked: List[Document] = []
        total_docs = 0
        total_chunks = 0

        for doc in tqdm(docs):
            total_docs += 1
            text = (doc.page_content or "").strip()
            if not text:
                logging.debug("Skipping empty document (source=%s)", getattr(doc, "metadata", {}).get("source"))
                continue

            try:
                chunks = splitter.split_text(text)
            except Exception as e:
                logging.exception("Failed to split document (source=%s): %s", getattr(doc, "metadata", {}).get("source"), e)
                # as a last resort, put the whole cleaned text as a single chunk
                chunks = [text]

            for i, c in enumerate(chunks):
                metadata = dict(doc.metadata or {})
                metadata["chunk_index"] = i
                # preserve a stable source field if not present
                if "source" not in metadata:
                    metadata["source"] = metadata.get("title") or metadata.get("file_name") or "unknown"
                chunked.append(Document(page_content=c, metadata=metadata))
                total_chunks += 1

        logging.info("Chunking complete: %d documents produced %d chunks (chunk_size=%s overlap=%s)",
                    total_docs, total_chunks, self.chunk_size, self.chunk_overlap)
        return chunked