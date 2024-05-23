import pickle
import streamlit as st

def question_answering_system(question):
    # LangChain components to use
    from langchain_community.vectorstores import Cassandra
    from langchain.indexes.vectorstore import VectorStoreIndexWrapper
    from langchain.llms import OpenAI
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.llms import OpenAI
    from langchain_core.output_parsers import StrOutputParser

    # Support for dataset retrieval with Hugging Face
    from datasets import load_dataset

    # With CassIO, the engine powering the Astra DB integration in LangChain,
    # you will also initialize the DB connection:
    import cassio
    from PyPDF2 import PdfReader
    
    ASTRA_DB_APPLICATION_TOKEN = "AstraCS:ExkWbcZsooJfhXiqYzyvtZBZ:4b1d07f8b327a444fe01c5a82c7643c5e588165613675a0c2af487db2d0274d8" # enter the "AstraCS:..." string found in in your Token JSON file
    ASTRA_DB_ID = "61fd00cb-13b1-414f-a661-6864332f9c7a" # enter your Database ID

    OPENAI_API_KEY = "sk-proj-FPcetihfTyoQQqOcJbQKT3BlbkFJq1rDLy9LflzxW8pfdtV6" # enter your OpenAI key

    # provide the path of  pdf file/files.
    pdfreader = PdfReader('testdata2.pdf')
    
    # read text from pdf
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="qa_mini_demo",
        session=None,
        keyspace=None,
    )

    from langchain.text_splitter import CharacterTextSplitter
    # We need to split the text using Character Text Split such that it should not increase token size
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    astra_vector_store.add_texts(texts[:50])

    print("Inserted %i headlines." % len(texts[:50]))

    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

    first_question = True
    answers = []

    # Process the question
    if question.lower() == "quit":
        return answers

    if question == "":
        return answers

    first_question = False

    # answers.append("\nQUESTION: \"%s\"" % question)s
    answer = astra_vector_index.query(question, llm=llm).strip()
    answers.append("ANSWER: \"%s\"\n" % answer)


    return answers

# Pickle the function along with required parameters
with open("question_answering_system.pkl", "wb") as f:
    pickle.dump(question_answering_system, f)

# Set up Streamlit UI
st.title("ParliamentaryPal")

# Input field for the question
question = st.text_input("Enter your question:")

# Button to get the answer
if st.button("Get Answer"):
    # Load the pickled function
    with open("question_answering_system.pkl", "rb") as f:
        question_answering_system = pickle.load(f)

    # Call the question answering system function
    answers = question_answering_system(question)

    # Display the answers in a box
    st.info("\n".join(answers)) 