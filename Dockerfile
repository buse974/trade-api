FROM node:20-slim
WORKDIR /app

# Python + pip for ML predict service
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN python3 -m pip install --break-system-packages xgboost psycopg2-binary numpy

# Node dependencies
COPY package*.json ./
RUN npm ci --only=production

COPY . .
EXPOSE 3000
CMD ["node", "src/index.js"]
