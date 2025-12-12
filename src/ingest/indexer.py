from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import os
from dotenv import load_dotenv
from pathlib import Path
from src.app.config import PERSIST_DIR, EMBEDDING_MODEL, logging
from typing import List, Optional, Dict, Any
from langchain_community.docstore.document import Document
from src.ingest.loader import Documents_loader
from src.ingest.chunker import Chunker
from src.retriever.vector_store import VectorStore


load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class Indexer:
    def __init__(
        self,
        uploaded_path: str,
        persist_dir: str,
        embedding_model: str,
        delete_after_index: bool = True,
    ):
        """
        uploaded_path: default path or directory used if not overridden when indexing
        persist_dir: where vector DB is stored
        embedding_model: embedding model name
        delete_after_index: whether to delete the uploaded file after indexing
        """
        self.uploaded_path = uploaded_path
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.delete_after_index = delete_after_index

    def index_file_to_vectorstore(self, uploaded_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Index a single uploaded file. If uploaded_path is provided, use it;
        otherwise fall back to self.uploaded_path.

        Returns a JSON-serializable summary dict.
        """
        target_path = uploaded_path or self.uploaded_path
        summary: Dict[str, Any] = {
            "file": str(target_path),
            "status": "ok",
            "pages_loaded": 0,
            "chunks_created": 0,
            "indexed_count": 0,
            "errors": [],
        }

        try:
            p = Path(target_path)
            if not p.exists():
                raise FileNotFoundError(f"Uploaded file missing: {target_path}")

            # ---------------------------
            # 1) Load doc(s) using your loader.
            # ---------------------------
            docs: List[Document] = []
            try:
                # Try constructing loader for a single file (many loaders accept a string path)
                loader = Documents_loader(str(p))

                # Preferred API: loader.load(path) or loader.load()
                if hasattr(loader, "load"):
                    try:
                        loaded = loader.load(str(p))
                    except TypeError:
                        # some loader.load() expects no args
                        loaded = loader.load()
                    if loaded is None:
                        docs = []
                    elif isinstance(loaded, list):
                        docs = loaded
                    else:
                        docs = [loaded]

                # Fallback API: load_all_docs
                elif hasattr(loader, "load_all_docs"):
                    try:
                        loaded = loader.load_all_docs(str(p))
                    except TypeError:
                        loaded = loader.load_all_docs()
                    if loaded is None:
                        docs = []
                    elif isinstance(loaded, list):
                        docs = loaded
                    else:
                        docs = [loaded]

                # Very defensive: if loader is callable
                elif callable(loader):
                    try:
                        loaded = loader(str(p))
                        if loaded is None:
                            docs = []
                        elif isinstance(loaded, list):
                            docs = loaded
                        else:
                            docs = [loaded]
                    except Exception:
                        docs = []

                else:
                    docs = []

            except Exception as e_single:
                # If constructing/using single-file loader fails, try directory-load fallback
                logging.debug("Single-file loader failed for %s: %s", p, e_single)
                try:
                    loader = Documents_loader(str(p.parent))
                    loaded = None
                    if hasattr(loader, "load_all_docs"):
                        try:
                            loaded = loader.load_all_docs()
                        except TypeError:
                            loaded = loader.load_all_docs()
                    elif hasattr(loader, "load"):
                        try:
                            loaded = loader.load()
                        except TypeError:
                            loaded = loader.load()
                    else:
                        # last resort: attempt to call
                        try:
                            loaded = loader()
                        except Exception:
                            loaded = None

                    if loaded is None:
                        docs = []
                    elif isinstance(loaded, list):
                        # filter to only documents originating from our file (if metadata exists)
                        docs = [
                            d
                            for d in loaded
                            if (d.metadata or {}).get("source") in (p.name, str(p))
                        ]
                    else:
                        # single document - check metadata
                        if (loaded.metadata or {}).get("source") in (p.name, str(p)):
                            docs = [loaded]
                        else:
                            docs = []
                except Exception as e_dir:
                    logging.exception("Directory loader fallback failed for %s: %s", p, e_dir)
                    docs = []

            # At this point 'docs' should be a list of Document objects
            if not docs:
                summary["status"] = "failed"
                summary["errors"].append(
                    "No documents/pages were loaded from file. Check loader API and file format."
                )
                logging.warning("Loader returned no documents for %s", str(p))
                return summary

            summary["pages_loaded"] = len(docs)
            logging.info("Loaded %d pages from %s", len(docs), p.name)

            # ---------------------------
            # 2) Chunk documents
            # ---------------------------
            chunker = Chunker()
            chunked_docs = chunker.chunk_documents(docs)
            summary["chunks_created"] = len(chunked_docs)
            logging.info("Chunked into %d chunks.", len(chunked_docs))

            if not chunked_docs:
                summary["status"] = "failed"
                summary["errors"].append("Chunker produced 0 chunks.")
                return summary

            # ---------------------------
            # 3) Index into vector store
            # ---------------------------
            vs_kwargs: Dict[str, Any] = {}
            if self.persist_dir:
                vs_kwargs["persist_dir"] = self.persist_dir
            if self.embedding_model:
                vs_kwargs["embedding_model"] = self.embedding_model

            vs = VectorStore(**vs_kwargs)

            existing_db = vs.load_vector_db()
            if existing_db:
                logging.info("Appending %d chunks to existing vector DB", len(chunked_docs))
                try:
                    existing_db.add_documents(chunked_docs)
                    try:
                        existing_db.save_local(PERSIST_DIR, index_name="faiss_index")
                    except Exception:
                        pass
                    summary["indexed_count"] = len(chunked_docs)
                except Exception as e:
                    logging.debug("existing_db.add_documents failed: %s", e)
                    # fallback to wrapper method
                    vs.add_documents(chunked_docs)
                    summary["indexed_count"] = len(chunked_docs)
            else:
                logging.info("No existing DB found — building new DB with %d chunks", len(chunked_docs))
                vs.build_db(chunked_docs)
                summary["indexed_count"] = len(chunked_docs)

            # ---------------------------
            # 4) Optionally delete uploaded file
            # ---------------------------
            if self.delete_after_index:
                try:
                    os.remove(p)
                    logging.info("Deleted uploaded file: %s", p)
                except Exception as e:
                    logging.warning("Unable to delete uploaded file %s: %s", p, e)

        except Exception as e_outer:
            logging.exception("Indexing failed for %s: %s", target_path, e_outer)
            summary["status"] = "failed"
            summary["errors"].append(str(e_outer))

        # Ensure file is a string for JSON serialization
        summary["file"] = str(summary.get("file", target_path))
        return summary


