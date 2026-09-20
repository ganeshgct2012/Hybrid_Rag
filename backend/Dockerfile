FROM python:3.11-slim


WORKDIR /app

# Install dependencies first, so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application code (no .env, no data)
#COPY app ./app
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]