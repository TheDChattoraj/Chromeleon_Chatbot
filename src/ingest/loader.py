import os
from langchain_community.document_loaders import PyPDFLoader
from src.app.config import logging


class Documents_loader:
    def __init__(self, files_dir: str):
        """
        files_dir: Path to directory containing documents.
        """
        self.files_dir = files_dir

    def load_all_docs(self):
        """
        Loads all .pdf documents from the given directory.
        Returns a list[Document].
        """
        all_docs = []

        if not os.path.exists(self.files_dir):
            logging.error(f"Documents directory not found: {self.files_dir}")
            return []

        file_list = os.listdir(self.files_dir)
        if not file_list:
            logging.warning(f"No files found in directory: {self.files_dir}")
            return []

        logging.info(f"Found {len(file_list)} files in {self.files_dir}")

        for fname in file_list:
            full_path = os.path.join(self.files_dir, fname)
            if not os.path.isfile(full_path):
                logging.debug(f"Skipping non-file: {full_path}")
                continue

            if fname.lower().endswith(".pdf"):
                try:
                    logging.info(f"Loading PDF: {fname}")
                    loader = PyPDFLoader(full_path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source"] = fname
                    all_docs.extend(docs)
                    logging.info(f"Loaded {len(docs)} pages from {fname}")
                except Exception as e:
                    logging.error(f"Failed to load {fname}: {e}")
            else:
                logging.debug(f"Skipping unsupported file: {fname}")

        logging.info(f"Total loaded docs: {len(all_docs)}")
        return all_docs