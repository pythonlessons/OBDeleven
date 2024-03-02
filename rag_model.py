import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


class RagChain:
    def __init__(
            self,
            data_path: str,
            chunk_size: int,
            chunk_overlap: int,
            openai_api_key: str,
            prompt_template: str,
            model_name: str,
            temperature: float,
            *args, **kwargs
        ):
        """
        Args:
            data_path (str): Path to the data file
            chunk_size (int): Size of the chunks
            chunk_overlap (int): Overlap between chunks
            openai_api_key (str): OpenAI API key
            prompt_template (str): Prompt template
            model_name (str): OpenAI model name
            temperature (float): OpenAI model temperature
        """
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", None)
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.temperature = temperature

        if not self.openai_api_key:
            raise ValueError("openai_api_key must be provided")

        # Load the data
        loader = TextLoader(self.data_path, encoding="utf-8")
        documents = loader.load()

        # Chunk the data
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.docs = text_splitter.split_documents(documents)

        # Load OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        # Setup FAISS vector database
        db = FAISS.from_documents(self.docs, self.embeddings)

        # Connect query to FAISS index using a retriever
        self.retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 4}
        )

        # Set the prompt template
        self.prompt_template = self.prompt_template or """
        You will be provided with questions and related data. 
        Your task is to find the answers to the questions using the given data. 
        If the data doesn't contain the answer to the question, then you must return 'Tokios informacijos nėra duomenų rinkinyje.'
        Question: {question} 
        Context: {context} 
        Answer:
        """

        # Create the prompt
        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        # Create the language model
        self.llm = ChatOpenAI(
            model_name=self.model_name, 
            temperature=self.temperature,
            openai_api_key=self.openai_api_key
            )

        # Create the RAG chain
        self.rag_chain = (
            {"context": self.retriever,  "question": RunnablePassthrough()} 
            | prompt 
            | self.llm
            | StrOutputParser() 
        )

    def invoke(self, user_message: str):
        assert isinstance(user_message, str), "user_message must be a string"

        return self.rag_chain.invoke(user_message)

    def __call__(self, user_message: str):
        return self.invoke(user_message)