FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime AS base

RUN apt-get update && rm -rf /var/lib/apt/lists/*

ENV CHECKPOINT_PATH=/checkpoints/latest.pth

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS test
COPY tests/requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt
COPY core/ core/
COPY serving/ serving/
COPY tests/ tests/
COPY pytest.ini .
RUN pytest -v


FROM base AS production
COPY serving/ serving/
COPY core/ core/

EXPOSE 8000

CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]