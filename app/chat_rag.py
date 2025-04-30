"""
Chat RAG functionality for the Streamlit app
"""

import streamlit as st
import time
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.core.query_engine import RouterQueryEngine
from db_config import get_pg_connection
from rag_utils import create_query_engine

def prepare_chat_messages(messages):
    """Convert Streamlit messages to LlamaIndex ChatMessage objects"""
    llama_messages = []
    
    for msg in messages:
        if msg["role"] == "user":
            llama_messages.append(ChatMessage(
                role=MessageRole.USER,
                content=msg["content"]
            ))
        elif msg["role"] == "assistant":
            llama_messages.append(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=msg["content"]
            ))
        elif msg["role"] == "system":
            llama_messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=msg["content"]
            ))
    
    return llama_messages

def should_use_rag(query):
    """Determine if a query would benefit from using RAG"""
    # Simple greetings and casual conversation should not trigger RAG
    simple_phrases = [
        "olá", "ola", "oi", "bom dia", "boa tarde", "boa noite", 
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "como vai", "tudo bem", "how are you", "what's up", "wassup", "what is up",
        "thank you", "thanks", "obrigado", "obrigada", "bye", "goodbye", "tchau",
        "yes", "no", "sim", "não", "nao"
    ]
    
    # Check for simple greetings
    if any(phrase == query.lower().strip() for phrase in simple_phrases):
        logging.info(f"Query '{query}' is a simple greeting - not using RAG")
        return False
    
    # Check if query is very short (less than 4 words) and doesn't contain 
    # keywords suggesting information retrieval
    word_count = len(query.split())
    info_keywords = ["what", "who", "how", "why", "when", "where", "explain", 
                    "describe", "tell", "show", "find", "search", "retrieve",
                    "o que", "quem", "como", "por que", "porque", "quando", "onde", 
                    "explique", "descreva", "diga", "mostre", "encontre", "busque"]
    
    if word_count < 4 and not any(keyword in query.lower() for keyword in info_keywords):
        logging.info(f"Query '{query}' is too short and lacks info-seeking keywords - not using RAG")
        return False
    
    # Default to using RAG for other queries
    return True

def rag_query_with_timeout(query_engine, user_query, timeout_seconds=30):
    """Run a RAG query with a timeout to prevent hanging"""
    result = {"done": False, "response": None, "error": None}
    
    def run_query():
        try:
            response = query_engine.query(user_query)
            result["response"] = response
            result["done"] = True
        except Exception as e:
            result["error"] = str(e)
            result["done"] = True
    
    # Start the query in a thread
    query_thread = threading.Thread(target=run_query)
    query_thread.daemon = True
    query_thread.start()
    
    # Wait for completion or timeout
    start_time = time.time()
    while not result["done"] and (time.time() - start_time) < timeout_seconds:
        time.sleep(0.1)
    
    if not result["done"]:
        return {"timeout": True, "error": f"A consulta excedeu o tempo limite de {timeout_seconds} segundos."}
    elif result["error"]:
        return {"timeout": False, "error": result["error"]}
    else:
        return {"timeout": False, "response": result["response"]}

def process_chat_input(model_id, use_rag=True, selected_files=None, selected_tables=None, similarity_top_k=3):
    """Process chat input using different methods based on selected options"""
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    user_query = st.chat_input("Sua pergunta:")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Check if we should use RAG (even if RAG is enabled)
            use_rag_for_this_query = use_rag and should_use_rag(user_query) and selected_tables and selected_files
            
            # Process based on selected mode
            if use_rag_for_this_query:
                response_container = st.container()
                
                with response_container:
                    message_placeholder.markdown("🧠 Pensando...")
                    
                    try:
                        # Display info about which docs are being used
                        st.markdown(f"📚 *Consultando: {', '.join(selected_files)}*")
                        
                        # Create response based on RAG
                        full_response = ""
                        
                        with st.spinner("Consultando documentos..."):
                            # Potential improvement: Use multiple tables and combine results
                            if not selected_tables:
                                raise ValueError("Nenhuma tabela selecionada para consulta")
                            
                            logging.info(f"Selected tables: {selected_tables}")
                            table_name = selected_tables[0]  # Start with just using the first table
                            
                            # Use our query engine creator
                            logging.info(f"Creating query engine for table {table_name} with model {model_id}")
                            query_engine = create_query_engine(
                                model_id, 
                                table_name, 
                                similarity_top_k=similarity_top_k
                            )
                            
                            # Get response from query engine with timeout
                            logging.info(f"Querying with: {user_query}")
                            result = rag_query_with_timeout(query_engine, user_query, timeout_seconds=45)
                            
                            if result.get("timeout"):
                                raise TimeoutError(result.get("error", "A consulta excedeu o tempo limite"))
                            
                            if result.get("error"):
                                raise Exception(result.get("error"))
                                
                            response = result.get("response")
                            
                            # Stream the response
                            if hasattr(response, 'response_gen'):
                                for text in response.response_gen:
                                    full_response += text
                                    message_placeholder.markdown(full_response + "▌")
                            else:
                                # Handle case where streaming isn't available
                                full_response = str(response)
                                message_placeholder.markdown(full_response)
                            
                            # Final response (removes cursor)
                            message_placeholder.markdown(full_response)
                            
                            # Display sources if available
                            if hasattr(response, 'source_nodes') and response.source_nodes:
                                with st.expander("📄 Fontes"):
                                    for i, source_node in enumerate(response.source_nodes):
                                        source_text = source_node.node.get_content()
                                        file_name = source_node.node.metadata.get("file_name", "Documento")
                                        st.markdown(f"**Fonte {i+1}** ({file_name}):")
                                        st.markdown(f"```\n{source_text[:500]}{'...' if len(source_text) > 500 else ''}\n```")
                    
                    except TimeoutError as e:
                        logging.error(f"RAG query timed out: {str(e)}")
                        message_placeholder.markdown(f"⏱️ **Tempo esgotado:** A consulta ao RAG demorou muito tempo. Usando o chat padrão.")
                        st.warning("Continuando com chat padrão sem RAG devido ao tempo limite excedido.")
                        full_response = process_standard_chat(user_query, model_id)
                        message_placeholder.markdown(full_response)
                    except Exception as e:
                        logging.error(f"Error in RAG processing: {str(e)}")
                        logging.exception("Detailed error:")
                        message_placeholder.markdown(f"❌ **Erro ao consultar base de conhecimento:** {str(e)}")
                        # Fallback to regular chat if RAG fails
                        st.warning("Continuando com chat padrão sem RAG devido ao erro.")
                        full_response = process_standard_chat(user_query, model_id)
                        message_placeholder.markdown(full_response)
            
            else:
                # Standard chat without RAG
                message_placeholder.markdown("🤖 Pensando...")
                full_response = process_standard_chat(user_query, model_id)
                message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

def process_standard_chat(query, model_id):
    """Process chat without using RAG, just direct LLM interaction"""
    try:
        # Get chat history
        messages = []
        
        # Add system message if not present
        system_message_present = any(msg["role"] == "system" for msg in st.session_state.messages)
        if not system_message_present:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content="Você é um assistente de IA útil e informativo."))
        
        # Add existing messages
        for msg in st.session_state.messages:
            role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
            messages.append(ChatMessage(role=role, content=msg["content"]))
        
        # Add the new query
        messages.append(ChatMessage(role=MessageRole.USER, content=query))
        
        # Create the LLM
        chat_llm = OpenAI(model=model_id, temperature=0.7)
        
        # Generate response
        logging.info(f"Sending standard chat request to model: {model_id}")
        response = chat_llm.chat(messages)
        return response.message.content
    except Exception as e:
        logging.error(f"Error in standard chat: {str(e)}")
        logging.exception("Detailed error:")
        return f"Desculpe, ocorreu um erro ao gerar a resposta: {str(e)}" 