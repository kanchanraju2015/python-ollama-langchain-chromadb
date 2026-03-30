from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Load the document
loader = TextLoader("state_of_the_union.txt",encoding="utf-8")# encoding must be done 
#note the file is in same directory 
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(
    separator="\\n\\n",
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.split_documents(documents)

# Print the first chunk as an example
print(docs[0].page_content)
