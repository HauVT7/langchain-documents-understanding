from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

model_file = "models"

# Load llm
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_file="llama",
        max_new_token="2048",
        temperature=0.01
    )



