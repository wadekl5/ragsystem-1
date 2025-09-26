import os
from openai import OpenAI

# from dotenv import find_dotenv,load_dotenv


# _=load_dotenv(find_dotenv())

client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import textwrap
def word_warp(text,width=80):
    return textwrap.fill(text,width)


from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter,SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder
import streamlit as st 



def extract_text(file):
    reader=PdfReader(file)
    extract_text=[p.extract_text() for p in reader.pages]
    extract_all_text=[text for text in extract_text if text]    
    return extract_all_text

def token_split(all_text):
    char_split=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n","\n","."," ",""]
    )

    split_text=char_split.split_text("\n\n".join(all_text))


    token_split=SentenceTransformersTokenTextSplitter(
        tokens_per_chunk=350,
        chunk_overlap=25
        )

    token_text_split=[]
    for text in split_text:
        token_text_split+=token_split.split_text(text)
    return token_text_split



ranked_text=[]
def reranking(query,retrived_text):
    cross=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks=cross.rank(query,retrived_text)
    for rank in ranks:
        doc=retrived_text[rank["corpus_id"]]
        ranked_text.append(doc)

def get_ans(query,ranked_text,model="gpt-3.5-turbo"):
    info="\n\n".join(ranked_text)
    messages=[
        {
            "role":"system",
            "content":"you are a helpful assistant , provide correct answer from the file"
        },
        {
            "role":"user",
            "content":f"fetch the answer from the given info:{info},\n Question:{query}"
        }
    ]
    
    response=client.chat.completions.create(
        model=model,
        messages=messages
    )

    content=response.choices[0].message.content
    return content


st.title("Rag System")

st.write("upload the file in pdf format")
upload_file=st.file_uploader(label="Upload Files",type=["pdf"])

if upload_file:
    text=extract_text(upload_file)
    token_text_split=token_split(text)

    chroma_client=chromadb.Client()

    if "user_docs" in [col.name for col in chroma_client.list_collections()]:
     chroma_client.delete_collection(name="user_docs")


    embedding_function=SentenceTransformerEmbeddingFunction()
    chroma_collection=chroma_client.get_or_create_collection(
            name="user_docs",
            embedding_function=embedding_function
    )



    ids=[str(f"{upload_file}_{i}") for i in range(len(token_text_split))]
    chroma_collection.add(ids=ids,documents=token_text_split)

    query=st.text_input(label="Question",placeholder="enter question here ")

    if query:
        result=chroma_collection.query(query_texts=[query],n_results=5,include=["documents"])
        retrived=result["documents"]
        retrived_text=[doc for i in retrived for doc in i]
        reranking(query,retrived_text)
       
        answer=get_ans(query=query,ranked_text=ranked_text)
        st.subheader("Anwer")
        st.write(word_warp(answer))
