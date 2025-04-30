FROM python:3.9-slim

WORKDIR /app

# Instala dependências de sistema necessárias + FERRAMENTAS DE REDE
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    dnsutils \
    iputils-ping \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Baixa e configura o Cloud SQL Auth Proxy (REMOVER SE NÃO FOR USAR SOCKET)
# Comentado pois estamos na Opção 2 (IP Público)
# RUN wget https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.15.2/cloud-sql-proxy.linux.amd64 -O /app/cloud-sql-proxy
# RUN chmod +x /app/cloud-sql-proxy

# Instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do projeto
COPY app/ app/
COPY start.sh .
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

# CMD ["./start.sh"] # CMD original comentado para teste

# CMD de teste de rede (rode APENAS PARA UM DEPLOY DE TESTE)
CMD ["bash", "-c", "echo '--- Iniciando Testes de Rede ---'; \
     echo '--- Teste DNS google.com ---'; dig google.com; \
     echo '--- Teste Ping google.com ---'; ping -c 4 google.com; \
     echo '--- Teste cURL google.com ---'; curl -v https://google.com; \
     echo '--- Teste cURL IP DB (telnet) ---'; curl -v --connect-timeout 10 telnet://34.48.95.143:5432; \
     echo '--- Testes Concluídos, dormindo... ---'; sleep 3600"]