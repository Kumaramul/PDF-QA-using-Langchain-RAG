from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from USEEmbeddings import USEEmbeddings

def run_retrieval_qa(text: str, question: str):
    # Split
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    
    # Embed
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embedding_model)
    
    # RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )
    
    result = qa.run(question)
    return result


def run_tf_retrieval_qa(llm, text: str, question: str):

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    docs = splitter.create_documents([text])

  
    embedding_model = USEEmbeddings()
    vectordb = FAISS.from_documents(docs, embedding_model)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 2}),
        chain_type="stuff"  #
    )

    return qa.run(question)

