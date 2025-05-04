from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory

embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def load_and_index(path="legal_dataset.txt"):
    loader = TextLoader(path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embedding)

db = load_and_index()
retriever = db.as_retriever()

qa_chain = ConversationalRetrievalChain.from_llm(llm=None, retriever=retriever, memory=memory)

def answer_query(question):
    return qa_chain.run(question)