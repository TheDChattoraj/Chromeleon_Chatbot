from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import os
from dotenv import load_dotenv
from src.app.config import PERSIST_DIR, EMBEDDING_MODEL, logging
from typing import List, Optional
from langchain_community.docstore.document import Document


load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")




class VectorStore:

    def __init__(self, persist_dir: str = PERSIST_DIR, embedding_model: str = EMBEDDING_MODEL):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model

        os.makedirs(self.persist_dir, exist_ok=True)

    def _create_embeddings(self):
        logging.info(f"Using embedding model: {self.embedding_model}")
        return OpenAIEmbeddings(model=self.embedding_model)

    def build_db(self, documents: List[Document]):
        """
        Build (or overwrite) a Chroma index from documents.
        """
        if not documents:
            raise ValueError("No documents provided to build the vector store.")

        logging.info(f"Building Chroma DB at: {self.persist_dir}")
        logging.info(f"Total documents/chunks received: {len(documents)}")

        embeddings = self._create_embeddings()

        db = FAISS.from_documents(documents, embeddings)

        db.save_local(folder_path= PERSIST_DIR, index_name= "faiss_index")
        logging.info("Chroma vector DB persisted successfully.")
        return db

    def load_vector_db(self):
        """
        Load existing Chroma DB if it exists.
        """
        if not os.path.exists(self.persist_dir):
            logging.warning(f"Persist directory does not exist: {self.persist_dir}")
            return None

        logging.info(f"Loading Chroma DB from: {self.persist_dir}")
        embeddings = self._create_embeddings()

        try:
            loaded_db = FAISS.load_local(folder_path= PERSIST_DIR, embeddings=embeddings, index_name="faiss_index", allow_dangerous_deserialization= True)
            logging.info("Vector database loaded successfully.")
            return loaded_db
        except Exception as e:
            logging.error(f"Failed to load Chroma DB: {e}")
            return None
        

    def add_documents(self, documents: List[Document]):
        """
        Append documents to an existing FAISS (or compatible) vector DB.

        Behavior:
        - If an existing DB is present, try multiple append APIs:
            1) db.add_documents(documents)
            2) db.add_texts(texts, metadatas=metadatas)
        - Attempt to persist the DB using common methods (persist, save_local).
        - If no DB exists, build a fresh one via build_db(documents).
        - If appending fails, we raise a clear error: rebuilding a FAISS index
        requires original documents (or an external store of original texts).
        """
        embeddings = self._create_embeddings()
        db = self.load_vector_db()

        # If no existing DB, create a new one
        if not db:
            logging.info("No existing FAISS DB found — building new DB.")
            return self.build_db(documents)

        # We have an existing DB: try to append
        try:
            # Preferred high-level API if present
            if hasattr(db, "add_documents"):
                logging.info("Using db.add_documents() to append %d docs.", len(documents))
                db.add_documents(documents)  # should use the DB's embedding function internally
            elif hasattr(db, "add_texts"):
                logging.info("Using db.add_texts() fallback to append %d docs.", len(documents))
                texts = [d.page_content for d in documents]
                metadatas = [d.metadata or {} for d in documents]
                # some implementations expect the embedding function passed at construction time
                db.add_texts(texts, metadatas=metadatas)
            else:
                # Last-ditch attempt: see if wrapper exposes 'from_documents' for merging (rare)
                raise AttributeError("Underlying FAISS vectorstore has no add_documents/add_texts API")

            # Persist the updated index if possible
            persisted = False
            try:
                if hasattr(db, "persist"):
                    db.persist()
                    persisted = True
                    logging.info("db.persist() succeeded.")
            except Exception as e:
                logging.debug("db.persist() failed or not supported: %s", e)

            try:
                # LangChain FAISS often exposes save_local(dir)
                if not persisted and hasattr(db, "save_local"):
                    db.save_local(self.persist_dir)
                    persisted = True
                    logging.info("db.save_local(%s) succeeded.", self.persist_dir)
            except Exception as e:
                logging.debug("db.save_local() failed: %s", e)

            # If neither method existed or succeeded, log a warning (but we did append in-memory)
            if not persisted:
                logging.warning(
                    "Appended documents to in-memory DB but could not detect or call a persistence method. "
                    "Make sure your DB is persisted to disk/storage to survive process restarts."
                )

            return db

        except Exception as e:
            # Appending failed: be explicit about the limitation
            logging.exception("Failed to append documents to FAISS DB: %s", e)

            # Rebuilding a FAISS index requires original document texts. If you don't have them,
            # the only safe option is to re-ingest from stored source files or fail fast.
            raise RuntimeError(
                "Appending to FAISS index failed. Note: rebuilding FAISS reliably requires the original "
                "documents or an external store of the documents' texts/metadata. "
                "If you keep originals, call build_db(all_documents) to rebuild. "
                "Original error: " + str(e)
            ) from e