FROM python:3.9-slim

WORKDIR /app

# Instala dependências de sistema necessárias
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Baixa e configura o Cloud SQL Auth Proxy (opcional, caso deseje usar no futuro)
RUN wget https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.15.2/cloud-sql-proxy.linux.amd64 -O /app/cloud-sql-proxy
RUN chmod +x /app/cloud-sql-proxy

# Instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do projeto
COPY app/ app/
COPY start.sh .
RUN chmod +x start.sh

# Configura a porta (usada por Streamlit e Cloud Run)
ENV PYTHONUNBUFFERED=1
EXPOSE 8501

# Inicia a aplicação via script (que também configura PYTHONPATH)
CMD ["./start.sh"]
