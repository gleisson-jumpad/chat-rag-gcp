# Base oficial compatível
FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Define porta usada pelo Streamlit
ENV PORT=8080

EXPOSE 8080

# Streamlit ouvindo na porta 8080 (requisito do Cloud Run)
CMD ["streamlit", "run", "app/main.py", "--server.port=8080", "--server.address=0.0.0.0"]
