AI-Powered Chatbot with Document Retrieval and Safety Filters
This project is a Flask-based web application featuring an AI-powered chatbot. The chatbot leverages Google Generative AI and LangChain to provide context-aware, accurate, and safe responses to user queries. The solution integrates document retrieval, safety mechanisms, and a conversational interface.

Features
AI-Powered Conversational Interface: Implements Google Generative AI to provide intelligent and natural language responses.
Document Retrieval: Retrieves contextually relevant information from uploaded documents using LangChain's Chroma vector store.
Safety Filters: Enforces safety guidelines to block responses promoting harmful content, hate speech, harassment, or misinformation.
Customizable Prompts: Utilizes a prompt template to align the chatbot's behavior with user requirements.
Session Management: Maintains conversation history using Flask sessions.
Web-Based Interface: User-friendly interface for easy interaction with the chatbot.
Technologies Used
Backend:

Flask: Python web framework for building the chatbot backend.
LangChain: Framework for managing prompts, retrieval, and QA pipelines.
Google Generative AI: For embeddings and natural language processing.
Storage:

Chroma: Vector database for storing and retrieving document embeddings.
Frontend:

HTML Templates: For rendering the web interface.
Other:

Recursive Text Splitter: To chunk documents for efficient processing.
Prerequisites
Python 3.8 or higher
Google Generative AI API Key
Required Python libraries (listed in requirements.txt)
Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/your-repo/chatbot-with-retrieval.git
cd chatbot-with-retrieval
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Set Up API Keys:

Replace google_api_key placeholders with your actual Google Generative AI API keys in the code.
Prepare Document:

Place your document file in the root directory with the name State_union.txt.
Usage
Run the Application:

bash
Copy code
python app.py
Access the Web Interface:

Open a browser and navigate to http://127.0.0.1:5000.
Ask Questions:

Enter your query in the input field and submit to receive context-aware answers.
Code Highlights
Document Processing:

Documents are split into manageable chunks using RecursiveCharacterTextSplitter for efficient vectorization and retrieval.
Prompt Template:

Customizable prompt template for defining the chatbot's behavior and response structure.
Safety Mechanisms:

Implements HarmBlockThreshold to block unsafe or inappropriate responses.
Retrieval-Augmented Generation:

Combines document retrieval with generative capabilities for accurate and contextually relevant answers.
File Structure
bash
Copy code
project/
├── app.py                # Main Flask application
├── State_union.txt       # Input document for retrieval
├── templates/
│   └── index.html        # Frontend template
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
Future Enhancements
Add user authentication for personalized experiences.
Extend support for multiple document formats (e.g., PDFs, Word files).
Enable real-time API monitoring and logging.
License
This project is licensed under the MIT License.

Acknowledgments
LangChain
Google Generative AI
