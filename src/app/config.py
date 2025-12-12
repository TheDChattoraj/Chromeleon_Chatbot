import logging
import os
from datetime import datetime
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate


FILES_PATH = "../Data Collection/Release Notes/"
TEST_FILES_PATH = "../Data Collection/Release Notes/test/"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

PERSIST_DIR = os.path.abspath(os.path.join(os.getcwd(), "vector_store", "faiss"))

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1"


PROMPT = """
        You are a helpful AI assistant for End User of Chromeleon Chromatographic Data System. Use the following context to answer the question at the end.
        If you don't know the answer, just say you don't know. Don't try to make up an answer.
        Context: {context}
        Question: {question}
        Answer:
        After answering, include a section called "Source" listing the PDF files used
    """


contextualized_q_system_prompt = (
    """
    Given a chat history and the latest user question, which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
)


#Logging cofiguration:
LOGFILE = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"

logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok= True)

LOG_FILE_PATH = os.path.join(logs_path, LOGFILE)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    filemode= 'w',
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s %(message)s",
    datefmt= '%d-%m-%Y_%H-%M-%S'
)