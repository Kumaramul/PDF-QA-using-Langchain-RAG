
from langchain_groq import ChatGroq
from pdf_extractor import download_pdf, extract_text_from_pdf
from db_retrieval import run_tf_retrieval_qa




if __name__ == "__main__":
    pdf_url = input("Enter PDF URL (online link): ")
    pdf_path = download_pdf(pdf_url)
    pdf_text = extract_text_from_pdf(pdf_path)

  
    llm = ChatGroq(
        groq_api_key="API_KEY",  # replace with your real key
        model_name="Gemma2-9b-It"
    )

    while True:
        question = input("Ask a question (or type 'exit'): ")
        if question.lower() == "exit":
            break
        answer = run_tf_retrieval_qa(llm, pdf_text, question)
        print("Answer:", answer)
