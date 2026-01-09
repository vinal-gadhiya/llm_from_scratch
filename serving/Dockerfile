FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

RUN apt-get update && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY serving/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]