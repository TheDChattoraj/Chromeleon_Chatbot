import argparse
import shutil
from pathlib import Path

from src.app.config import TEST_FILES_PATH, logging, PERSIST_DIR
from src.ingest.loader import Documents_loader
from src.ingest.chunker import Chunker
from src.retriever.vector_store import VectorStore

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x



def build_vector_store(rebuild: bool = False):
    tests_path = Path(TEST_FILES_PATH)
    persist_dir = Path(PERSIST_DIR)

    if not tests_path.exists() or not tests_path.is_dir():
        logging.error("TEST_FILES_PATH %s does not exist or is not a directory", tests_path)
        return

    # if rebuild requested, remove existing persist dir
    if rebuild and persist_dir.exists():
        logging.info("Rebuild requested — removing existing persist dir: %s", persist_dir)
        try:
            shutil.rmtree(persist_dir)
        except Exception as e:
            logging.exception("Failed to remove persist dir: %s", e)
            return

    # initialize helpers
    logging.info("Initializing loader, chunker and vector store")
    loader = Documents_loader(str(tests_path))
    chunker = Chunker()
    vector_store = VectorStore(persist_dir=str(persist_dir))

    # if DB already exists and not rebuilding, load and skip
    existing = vector_store.load_vector_db()
    if existing and not rebuild:
        logging.info("Found existing vector DB at %s — skipping build (use --rebuild to force)", persist_dir)
        return

    logging.info("Loading documents from %s", tests_path)
    try:
        # Most loader implementations offer a no-arg load_all_docs() that uses configured path
        docs = loader.load_all_docs()
    except TypeError:
        # fallback in case loader requires a list argument
        docs = loader.load_all_docs([str(tests_path)])

    if not docs:
        logging.error("No documents returned by loader. Nothing to index.")
        return

    logging.info("Loaded %d document(s) / pages. Chunking...", len(docs))
    chunked_docs = chunker.chunk_documents(docs)
    logging.info("Chunking produced %d chunks.", len(chunked_docs))

    if not chunked_docs:
        logging.error("Chunker returned zero chunks. Aborting.")
        return

    # Build (overwrite) the DB from chunked docs
    logging.info("Building vector DB (this may take some time)...")
    try:
        db = vector_store.build_db(chunked_docs)
        logging.info("Vector DB built and persisted to: %s", persist_dir)
    except Exception as e:
        logging.exception("Failed to build vector DB: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Minimal bootstrap: build vector store from TEST_FILES_PATH")
    parser.add_argument("--rebuild", action="store_true", help="Remove existing persist dir and rebuild from scratch")
    args = parser.parse_args()

    build_vector_store(rebuild=args.rebuild)


if __name__ == "__main__":
    main()

