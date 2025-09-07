from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

def run_simple_qa(llm, text: str, question: str):
    docs = [Document(page_content=text)]
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=question)
    return response