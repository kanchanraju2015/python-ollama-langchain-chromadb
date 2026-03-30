
# first install chroma db as pip install chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma  # for vector store 

from langchain_ollama import OllamaEmbeddings
#embedding=Ollamaembeddings(model="gemma:2b")  # LLM model
embeddings=OllamaEmbeddings(model="gemma:2b") # this is must be running http://localhost:11434


text=["alpha is the first letter of greek alphabet.",

      "beta is the second letter of greek alphabet"]

vectors=embeddings.embed_documents(text)  # this is embedded


# embed a query 

query="what is the first greek alphabet"   
query_vector=embeddings.embed_query(query) # this is query on behalf of embeded text

from langchain_community.vectorstores import Chroma 

db=Chroma.from_texts(text,embeddings) # saves to the chroma db 

results=db.similarity_search(query,k=1) # top 1 similarity search 

print(results[0].page_content)


#pip install -U `langchain-ollama`
#from `langchain_ollama import OllamaEmbeddings``.

