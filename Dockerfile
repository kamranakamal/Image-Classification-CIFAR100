FROM python:3.8-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install CPU-only PyTorch with retry logic and extended timeout
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1 torchvision==0.15.2

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["python", "app/gradio_app.py"]