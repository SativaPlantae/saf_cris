import streamlit as st
import os
import pandas as pd
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# 🔐 Chave da OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

# 🔧 Função para carregar e limpar automaticamente o data.csv
def carregar_e_limpar_dados(caminho_csv: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv, sep=";")

    def limpar_moeda(valor):
        if isinstance(valor, str):
            valor = valor.replace("R$", "").replace(".", "").replace(",", ".").strip()
            try:
                return float(valor)
            except:
                return valor
        return valor

    colunas_monetarias = [
        "Faturamento anual", "Despesas anuais", "Lucro anual", "Preço de venda"
    ]

    for coluna in colunas_monetarias:
        if coluna in df.columns:
            df[coluna] = df[coluna].apply(limpar_moeda)

    def separar_valor_unidade(valor):
        if isinstance(valor, str):
            match = re.match(r"([\d,\.]+)\s*(\w+)", valor.strip())
            if match:
                valor_numerico = match.group(1).replace(",", ".")
                unidade = match.group(2)
                return float(valor_numerico), unidade
        return None, None

    if "Produção por indivíduo (kg, un ou m³)" in df.columns:
        df["Producao_individual_valor"], df["Producao_individual_unidade"] = zip(
            *df["Produção por indivíduo (kg, un ou m³)"].map(separar_valor_unidade)
        )
        df.drop(columns=["Produção por indivíduo (kg, un ou m³)"], inplace=True)

    return df

# 🧠 Cadeia com memória
@st.cache_resource
def carregar_chain_com_memoria():
    df = carregar_e_limpar_dados("data.csv")
    texto_unico = "\n".join(df.astype(str).apply(lambda x: " | ".join(x), axis=1))
    document = Document(page_content=texto_unico)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents([document])

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # ✅ Diretório persistente necessário para Chroma
    persist_directory = "chroma_db"
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    retriever = vectorstore.as_retriever()

    prompt_template = PromptTemplate(
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

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    return chain

# 🌱 Interface visual
st.set_page_config(page_title="Chatbot SAF Cristal 🌱", page_icon="🐝")
st.title("🐝 Chatbot do SAF Cristal")
st.markdown("Converse com o assistente sobre o Sistema Agroflorestal Cristal 📊")

# 🧹 Botão para limpar conversa
if st.button("🧹 Limpar conversa"):
    st.session_state.mensagens = []
    st.experimental_rerun()

# Inicializa estado
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = carregar_chain_com_memoria()

# Exibir histórico da conversa
for remetente, mensagem in st.session_state.mensagens:
    with st.chat_message("user" if remetente == "🧑‍🌾" else "assistant", avatar=remetente):
        st.markdown(mensagem)

# Campo de entrada
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
