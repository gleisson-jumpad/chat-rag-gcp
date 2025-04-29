import streamlit as st

st.set_page_config(page_title="Chat RAG", page_icon="🤖")

st.title("Chat RAG - Beta 🚀")
st.write("Olá! Esta é a primeira versão do nosso chat com RAG no GCP!")

# Chat simple placeholder
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input de usuário
if user_input := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Responder com placeholder
    response = f"Você disse: {user_input}"
    st.session_state.messages.append({"role": "assistant", "content": response})
