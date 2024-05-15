from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# Useful to add documents to the chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Useful to load the URL into documents
from langchain_community.document_loaders import WebBaseLoader
# Split the Web page into multiple chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Create Embeddings
from langchain_openai import OpenAIEmbeddings
# Vector Database FAISS
from langchain_community.vectorstores.faiss import FAISS
# USeful to create the Retrieval part
from langchain.chains import create_retrieval_chain

"""**Setting the Context from a Web Page**"""

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(api_key=openaikey, model_name="gpt-3.5-turbo", temperature = 0.5)
response = llm.invoke("What is LCEL")
print(response)

"""The above assumes the meaning of LCEL , but we want to get the LLM to answer from the following Web page

[LCEL](https://python.langchain.com/docs/expression_language/)

**HOW DO WE WRITE CODE to do that ?**

**HELPER FUNCTIONS**

**RETRIEVE DATA FROM WEBPAGE**
"""

# Retrieve Data
def get_docs():
    loader = WebBaseLoader('https://python.langchain.com/docs/expression_language/')
    docs = loader.load()

    # WE need to split the web page data
    # We create chunks of 200 and overlap so no data is missed out
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    splitDocs = text_splitter.split_documents(docs)

    return splitDocs

"""**Create a Vector Database**"""

def create_vector_store(docs):
    embedding = OpenAIEmbeddings(api_key=openaikey)
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

"""**Putting all together**"""

def create_chain(vectorStore):
    model = ChatOpenAI(api_key=openaikey,
        temperature=0.4,
        model='gpt-3.5-turbo-1106'
    )
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question.
    Context: {context}
    Question: {input}
    """)

    print(prompt)

    # chain = prompt | model
    # We are creating the chain to add documents
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    # Retrieving the top 1 relevant document from the vector store , We can change k to 2 and get top 2 and so on
    retriever = vectorStore.as_retriever(search_kwargs={"k": 1})

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

"""**Using the Context from the Website - Changes the response with the new context**"""

docs = get_docs()
vectorStore = create_vector_store(docs)
chain = create_chain(vectorStore)

response = chain.invoke({
    "input": "What is LCEL?",
})
