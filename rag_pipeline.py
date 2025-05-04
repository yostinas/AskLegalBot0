from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Load embedding model
embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Set up memory to preserve chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load and index the documents
def load_and_index(path="legal_dataset.txt"):
    loader = TextLoader(path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embedding)

# Load DB and initialize retriever
db = load_and_index()
retriever = db.as_retriever()

# Setup the QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=None,  # You will need to provide a language model here if using one
    retriever=retriever,
    memory=memory
)

# Run the query through the QA system
def answer_query(question):
    return qa_chain.run(question)

# Reload documents if updated
def update_documents(path):
    global db, retriever, qa_chain
    db = load_and_index(path)
    retriever = db.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=None,
        retriever=retriever,
        memory=memory
    )
