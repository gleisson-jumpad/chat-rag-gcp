FROM python:3.9-slim

WORKDIR /app

# Instala dependências de sistema necessárias
# REMOVIDO ferramentas de rede extras, pois não são mais necessárias para produção
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    # dnsutils e iputils-ping removidos para imagem menor
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cloud SQL Auth Proxy comentado (correto para Opção 2)
# RUN wget ...
# RUN chmod ...

# Instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do projeto
COPY app/ app/
COPY start.sh .
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

# --- CORREÇÃO DO CMD ---
# Comenta ou remove o CMD de teste de rede:
# CMD ["bash", "-c", "echo '--- Iniciando Testes de Rede ---'; ... sleep 3600"]

# Descomenta e ativa o CMD original para iniciar a aplicação:
CMD ["./start.sh"]
# ----------------------