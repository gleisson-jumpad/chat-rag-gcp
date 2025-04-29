# Chat RAG GCP

A RAG (Retrieval-Augmented Generation) chat system hosted on Google Cloud Platform.

## Overview

This project implements a chat system with intelligent file upload capabilities, using:

- **Backend**: Python 3.11+
- **Database**: PostgreSQL with pgvector for embeddings
- **Retrieval & Vectorization**: LlamaIndex
- **LLM**: OpenAI GPT-4o-mini
- **Frontend**: Streamlit/Gradio

## Features

- Web chat interface with modern design
- Secure file upload with extension validation
- Automatic vectorization of uploaded files
- Dynamic RAG via tool-calling
- Session-based document management

## Technical Architecture

- Files uploaded are immediately vectorized and stored in PostgreSQL
- LLM can request relevant document retrieval through function calling
- Each chat session has isolated document context

## Setup

Instructions for setting up the project will be added here.

## Development

This project is under active development. 