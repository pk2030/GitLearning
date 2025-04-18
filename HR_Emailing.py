import streamlit as st
import os
import PyPDF2
from langchain_openai import AzureChatOpenAI

AZURE_OPENAI_API_KEY = "0ba58e88a4264c94a2eef6a06940a412"
AZURE_OPENAI_ENDPOINT = "https://evokerpaai.openai.azure.com/"
DEPLOYMENT_NAME = "RPAAI"
API_VERSION = "2024-05-01-preview"

MAX_TOKENS = 2000
SEED = 42
TOP_P = 1.0
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0

REFERENCE_FOLDER = "./reference"
os.makedirs(REFERENCE_FOLDER, exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Saves uploaded file to the reference folder."""
    file_path = os.path.join(REFERENCE_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def list_pdf_files():
    """Lists PDF files in the reference folder."""
    return [f for f in os.listdir(REFERENCE_FOLDER) if f.endswith(".pdf")]

def load_reference_email(file_name):
    """Loads reference email content from a selected PDF file."""
    file_path = os.path.join(REFERENCE_FOLDER, file_name)
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def generate_emails(content, prompt, num_emails, email_type, sample_email, temperature):
    """Generates multiple emails in a single LLM call."""
    system_prompt = f"""
Generate {num_emails} unique {email_type.lower()} emails.

Content: {content}
Instructions: {prompt}
Sample Email: {sample_email}

Separate each email with '---'.
Strictly do not include any names or personal information or company details in mail.
"""
    
    messages = [
        {"role": "system", "content": "You are an AI that generates engaging, professional, and structured emails."},
        {"role": "user", "content": system_prompt}
    ]
    
    llm = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=DEPLOYMENT_NAME,
        api_version=API_VERSION,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        seed=SEED,
        top_p=TOP_P,
        presence_penalty=PRESENCE_PENALTY,
        frequency_penalty=FREQUENCY_PENALTY
    )
    
    response = llm.invoke(messages)
    email_texts = response.content.split('---')
    return [email.strip() for email in email_texts if email.strip()]

def main():
    st.set_page_config(page_title="📨 AI Email Generator", layout="wide")
    st.markdown("<h1 style='font-size:30px; margin-bottom:10px;'><center>📨 AI-Powered Email Generator</center></h1>", unsafe_allow_html=True)
    
    st.sidebar.title("⚙️ Settings")
    
    content = st.sidebar.text_area("📄 **1. Content***", "", help="Enter the core message of your email")
    prompt = st.sidebar.text_area("🎨 **2. Instruction***", "", help="Define the tone or style of the email")
    email_type = st.sidebar.selectbox("📜 **3. Type**", ["Normal Mail", "Reply to Mail"])
    
    pdf_files = list_pdf_files()
    reference_mail = st.sidebar.selectbox("📂 **4. Reference Mails**", ["Custom"] + pdf_files, help="Choose a reference email category")
    
    sample_email = ""
    if reference_mail == "Custom":
        uploaded_files = st.sidebar.file_uploader("📤 Upload PDF References", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_uploaded_file(uploaded_file)
            st.sidebar.success("✅ Files uploaded successfully!")
        
        if sample_email.strip():
            with open(os.path.join(REFERENCE_FOLDER, "custom_sample.txt"), "w") as f:
                f.write(sample_email)
    else:
        sample_email = load_reference_email(reference_mail)
        st.sidebar.text_area("📌 **6. Sample Email (Optional)**", sample_email, height=200)
    
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    num_emails = st.sidebar.slider("📬 **7. Number of Emails**", 1, 10, 4)
    temperature = st.sidebar.slider(" **8. Temperature**", 0.0, 1.0, 0.5, help="Adjust the creativity level of the AI-generated emails")
    
    generate_button = st.sidebar.button("🚀 Generate Emails", use_container_width=True)
    
    if generate_button:
        if content and prompt:
            emails = generate_emails(content, prompt, num_emails, email_type, sample_email, temperature)
            st.sidebar.success("✅ Emails Generated Successfully!")
            
            for i, email in enumerate(emails):
                st.markdown(f"<h3 style='font-size:16px; margin-bottom:10px;'>Email {i+1}</h3>", unsafe_allow_html=True)
                st.code(email, language="markdown", line_numbers=False, wrap_lines=True)
                st.markdown("---")
        else:
            st.sidebar.warning("⚠️ Please provide both Email Content and Writing Style.")

if __name__ == "__main__":
    main()
