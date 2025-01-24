FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Optionally set defaults for environment variables
ENV ONE_MIN_AVAILABLE_MODELS "mistral-nemo,gpt-4o,deepseek-chat"
ENV PERMIT_MODELS_NOT_IN_AVAILABLE_MODELS "False"

EXPOSE 5001

CMD ["python", "main.py"]