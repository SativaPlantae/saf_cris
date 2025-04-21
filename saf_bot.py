import os
import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

st.write("🔐 API KEY detectada?" , bool(openai_api_key))

# Chave da OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def carregar_chain_com_memoria():
    df = pd.read_csv("data.csv", sep=";")
    texto_unico = "\n".join(df.astype(str).apply(lambda x: " | ".join(x), axis=1))
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(
        [Document(page_content=texto_unico)]
    )

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template="""
Você é um assistente virtual treinado com base em uma planilha técnica sobre o Sistema Agroflorestal SAF Cristal.

Fale de forma clara, didática e acessível, como se estivesse conversando com um estudante ou alguém curioso. Use o histórico da conversa para manter a fluidez. Evite respostas robóticas. Se não tiver certeza, diga isso de forma sutil e humana.

-------------------
Histórico:
{chat_history}

Informações encontradas:
{context}

Pergunta: {question}
Resposta:"""
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0.5, openai_api_key=openai_api_key),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# Interface do app
st.set_page_config(page_title="Chatbot SAF Cristal 🌱", page_icon="🐝")
st.title("🐝 Chatbot do SAF Cristal")
st.markdown("Converse com o assistente sobre o Sistema Agroflorestal Cristal 📊")

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = carregar_chain_com_memoria()

for remetente, mensagem in st.session_state.mensagens:
    with st.chat_message("user" if remetente == "🧑‍🌾" else "assistant", avatar=remetente):
        st.markdown(mensagem)

user_input = st.chat_input("Digite sua pergunta aqui...")

if user_input:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_input)
    st.session_state.mensagens.append(("🧑‍🌾", user_input))

    with st.spinner("Consultando o SAF Cristal..."):
        try:
            resposta = st.session_state.qa_chain.run(user_input)
        except Exception as e:
            resposta = f"⚠️ Ocorreu um erro: {e}"

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)
    st.session_state.mensagens.append(("🐝", resposta))
