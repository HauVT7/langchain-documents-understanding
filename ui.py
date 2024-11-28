import gradio as gr
import os
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from accelerate import Accelerator

# Configuration
llm_model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vector-store/db-faiss"
embedding_model_file = "models/all-MiniLM-L6-v2-f16.gguf"


# Load llm
def load_llm(model_file):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    n_gpu_layers = 40
    n_batch = 512
    # llm = CTransformers(
    #     model=model_file,
    #     model_file="llama",
    #     max_new_token="2048",
    #     temperature=0.01,
    #     n_gpu_layer=n_gpu_layers,
    #     n_batch=n_batch,
    #     device=1
    # )
    accelerator = Accelerator()
    config = {'max_new_tokens': 512, 'temperature': 0.01}
    llm = CTransformers(model=model_file, model_type="llama", gpu_layers=50, config=config)
    llm, config = accelerator.prepare(llm, config)
    return llm


# Create prompt template
def create_prompt_template(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt


# Read data from vector db
def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(
        model_file=embedding_model_file,
        gpt4all_kwargs={'allow_download': 'True'},
        device="cuda"
    )
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db


# Create QA chain
def create_qa_chain(prompt, llm, vector_db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain


# Load vector db and llm
db = read_vectors_db()
llm = load_llm(llm_model_file)

# Generate prompt
vinallama_prompt_template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. 
Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = create_prompt_template(vinallama_prompt_template)

# Create QA chain
chain = create_qa_chain(prompt, llm, db)

# Function for Gradio to handle queries
# Function for Gradio to handle queries and format the output for proper line breaks
# Function for Gradio to handle queries and format the output using Markdown
def qa_system(query):
    response = chain.invoke({"query": query})

    cleaned_response = response['result'].replace('\n', '\n\n').strip()

    return cleaned_response


# Create Gradio interface
iface = gr.Interface(
    fn=qa_system,
    inputs=gr.Textbox(label="Enter your question"),
    outputs="text",
    title="Document Understanding",
    description="Ask any question related to the products and get an answer from the knowledge base."
)

# Launch the app
iface.launch()
