#!/usr/bin/env python3
"""
Improved RAG Processor module that directly uses the enhanced MultiTableRAGTool query method
"""
import os
import logging
import openai
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_processor")

def process_query(user_message: str, rag_tool: Any, model: str = "gpt-4o") -> str:
    """
    Process a user query using the improved MultiTableRAGTool
    
    This function is a simplified replacement for process_message_with_selective_rag
    that directly uses our enhanced query method with financial and contract information detection
    
    Args:
        user_message: The user's query
        rag_tool: Instance of MultiTableRAGTool
        model: The OpenAI model to use
        
    Returns:
        Formatted response with answer and sources
    """
    logger.info(f"Processing query: '{user_message}'")
    
    # Ensure the OpenAI API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return "ERROR: OpenAI API key not set in environment variables"
    
    # Get information about available files
    files_info = rag_tool.get_files_in_database()
    available_docs = files_info.get("all_files", [])
    
    logger.info(f"Found {len(available_docs)} documents in the database")
    
    try:
        # Directly use our improved query method with financial and contract detection
        logger.info("Using enhanced MultiTableRAGTool.query for processing")
        result = rag_tool.query(user_message, model=model)
        
        # Check for errors in the result
        if "error" in result:
            logger.error(f"Error in query result: {result['error']}")
            return f"Não foi possível encontrar informações relevantes: {result.get('message', 'Erro desconhecido')}"
        
        # Get the answer
        answer = result.get("answer", "")
        
        # If answer is empty, return a helpful message
        if not answer or answer.strip() == "":
            logger.warning("Empty answer returned from query")
            return "Não consegui encontrar informações relevantes nos documentos para responder essa pergunta."
            
        logger.info(f"Successfully generated response of length {len(answer)}")
        return answer
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.exception("Detailed traceback:")
        return f"Ocorreu um erro ao processar sua consulta: {str(e)}"

def process_query_with_llm(user_message: str, rag_tool: Any, model: str = "gpt-4o") -> str:
    """
    Process a user query using the improved MultiTableRAGTool and then enhance the response with an LLM
    
    This is a more sophisticated version that first gets RAG results and then uses the LLM to create
    a more refined response based on those results.
    
    Args:
        user_message: The user's query
        rag_tool: Instance of MultiTableRAGTool
        model: The OpenAI model to use
        
    Returns:
        Formatted response with answer and sources
    """
    logger.info(f"Processing query with LLM enhancement: '{user_message}'")
    
    # Ensure the OpenAI API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return "ERROR: OpenAI API key not set in environment variables"
    
    try:
        # First get the RAG results
        logger.info("Getting RAG results")
        result = rag_tool.query(user_message, model=model)
        
        # Check for errors
        if "error" in result:
            logger.error(f"Error in query result: {result['error']}")
            return f"Não foi possível encontrar informações relevantes: {result.get('message', 'Erro desconhecido')}"
        
        # Extract answer and source information
        rag_answer = result.get("answer", "")
        sources = result.get("sources", [])
        
        # If no meaningful answer, return a helpful message
        if not rag_answer or rag_answer.strip() == "":
            logger.warning("Empty answer returned from query")
            return "Não consegui encontrar informações relevantes nos documentos para responder essa pergunta."
        
        # If we have sources, create a more refined response using the LLM
        if sources:
            logger.info(f"Found {len(sources)} sources, enhancing response with LLM")
            
            # Create system prompt with enhanced instructions
            system_prompt = """
            Você é um assistente especializado em responder perguntas com base em informações extraídas de documentos.
            Analise cuidadosamente as informações e responda de forma clara e concisa.
            
            Diretrizes:
            - Responda APENAS com informações encontradas nos documentos
            - Seja preciso e objetivo
            - Não invente ou infira informações além das fornecidas
            - Para valores, números e datas, cite exatamente como aparecem nos documentos
            - Para nomes de pessoas e organizações, mantenha a grafia original
            """
            
            # Format the source information
            source_info = ""
            for i, source in enumerate(sources[:5], 1):  # Limit to top 5 sources
                doc_name = source.get("metadata", {}).get("file_name", "Unknown document")
                text = source.get("text", "")
                source_info += f"\nExcerto {i} (do documento '{doc_name}'):\n{text}\n"
            
            # Create the user message with RAG results
            user_prompt = f"""
            Pergunta do usuário: {user_message}
            
            Informações encontradas nos documentos:
            {source_info}
            
            Baseando-se APENAS nas informações acima, responda à pergunta de forma clara e direta.
            """
            
            # Get enhanced response from LLM
            logger.info("Generating enhanced response with LLM")
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            enhanced_answer = response.choices[0].message.content
            logger.info(f"Successfully generated enhanced response of length {len(enhanced_answer)}")
            
            return enhanced_answer
        
        # If no sources but we have an answer, just return the RAG answer
        logger.info("No sources found, returning direct RAG answer")
        return rag_answer
        
    except Exception as e:
        logger.error(f"Error processing query with LLM: {str(e)}")
        logger.exception("Detailed traceback:")
        return f"Ocorreu um erro ao processar sua consulta: {str(e)}" 