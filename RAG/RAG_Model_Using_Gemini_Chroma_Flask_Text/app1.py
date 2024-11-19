from flask import Flask, request, jsonify, render_template, session
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import os

app = Flask(__name__)

app.secret_key = os.urandom(24)

# Initialize components
def initialize_qa_system():
    # Load documents
    loader = TextLoader('State_union.txt', encoding='utf-8')
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Set up embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key='AIzaSyDyi6QYcIAU4kQy6pFTUl0MeIKk6Sel_FA',
        task_type="retrieval_query"
    )

    # Create the vector store
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings)

    # Define the prompt template
    prompt_template = """
    ## Safety and Respect Come First!

    You are programmed to be a helpful and harmless AI. You will not answer requests that promote:

    * **Harassment or Bullying:** Targeting individuals or groups with hateful or hurtful language.
    * **Hate Speech:**  Content that attacks or demeans others based on race, ethnicity, religion, gender, sexual orientation, disability, or other protected characteristics.
    * **Violence or Harm:**  Promoting or glorifying violence, illegal activities, or dangerous behavior.
    * **Misinformation and Falsehoods:**  Spreading demonstrably false or misleading information.

    **How to Use You:**

    1. **Provide Context:** Give me background information on a topic.
    2. **Ask Your Question:** Clearly state your question related to the provided context.

    ##  Answering User Question:
    Context: \n {context}
    Question: \n {question}
    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    # Set up safety settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    # Set up the chat model with temperature
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key='AIzaSyBk3PYxbPaNiSrwdlvsTJS4uAoPCp_Y_mo',
        temperature=0.7,
        safety_settings=safety_settings
    )

    # Create the QA chain
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        llm=chat_model
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever_from_llm,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# Initialize QA chain globally
qa_chain = initialize_qa_system()

# Function to maintain conversation history in session
def get_conversation_history():
    if 'conversation' not in session:
        session['conversation'] = []
    return session['conversation']

def update_conversation_history(user_question, bot_response):
    conversation = get_conversation_history()
    conversation.append({"user": user_question, "bot": bot_response})
    session['conversation'] = conversation

# Define the main chatbot route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the user's query
        user_question = request.form["question"]

        # Prepare the context (you can adjust this as needed)
        context = "document context here"  # Replace with actual logic to retrieve document context

        # Run the query through the QA chain
        response = qa_chain.invoke({"query": user_question, "context": context})

        # Extract the answer from the response
        # Assuming response has 'result' and 'source_documents' attributes
        bot_response = response['result']  # Get the generated answer
        
        # Optional: If you want to include source documents, handle them here
        source_documents = response.get('source_documents', [])
        source_texts = [doc.page_content for doc in source_documents]  # Extract the text content if needed

        # Display the response on the webpage
        return render_template("index.html", user_question=user_question, bot_response=bot_response, source_documents=source_texts)

    return render_template("index.html", user_question=None, bot_response=None)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
