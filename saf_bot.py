import os
import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# 🔐 Chave da OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def carregar_qa_chain():
    # 1. Carregando os dados do SAF Cristal
    df = pd.read_csv("data.csv")
    texto_unico = "\n".join(df.astype(str).apply(lambda x: " | ".join(x), axis=1))

    # 2. Transformando em documento LangChain
    document = Document(page_content=texto_unico)

    # 3. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents([document])

    # 4. Vetorização
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # 5. Prompt mais natural, sem repetição robótica
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Você é um assistente virtual treinado com base em uma planilha de dados técnicos sobre o Sistema Agroflorestal SAF Cristal.

Sua linguagem é didática, amigável e acessível, como se estivesse conversando com um estudante ou profissional da área ambiental. Evite repetições, formalismos excessivos e frases robóticas.

Caso não tenha certeza da resposta, seja transparente e diga isso de forma leve, como:
- "Não tenho certeza absoluta, mas posso te contar o que aparece por aqui."
- "Parece que esse detalhe não está explícito, mas vamos lá..."
- "Não encontrei exatamente isso, mas com base no que sei, diria que..."

-------------------
{context}

Pergunta: {question}
Resposta:"""
    )

    # 6. Modelo LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=500,
        openai_api_key=openai_api_key
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa_chain

# 🌐 Interface do app
st.set_page_config(page_title="Chatbot SAF Cristal - Sativa Plantae", page_icon="🌬")
st.title("🌬 Chatbot do SAF Cristal")
st.markdown("Converse com o assistente sobre o Sistema Agroflorestal Cristal 📊")

# Histórico de conversa
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

qa_chain = carregar_qa_chain()

# Formulário de envio
with st.form(key="formulario_chat", clear_on_submit=True):
    user_input = st.text_input("🧑‍🌾 Você:", placeholder="Pergunte algo sobre o SAF Cristal...")
    submit = st.form_submit_button("Enviar")

# Processamento
if submit and user_input:
    with st.spinner("Consultando o SAF..."):
        try:
            resposta = qa_chain.run(user_input)
            st.session_state.mensagens.append(("🧑‍🌾", user_input))
            st.session_state.mensagens.append(("🌬", resposta))
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")

# Exibição da conversa completa
for remetente, mensagem in st.session_state.mensagens:
    st.markdown(f"**{remetente}**: {mensagem}")
