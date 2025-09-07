import requests
import pdfplumber

#  Download PDF from URL
def download_pdf(url: str, output_path: str = "temp.pdf"):
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    return output_path


#  Extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text
