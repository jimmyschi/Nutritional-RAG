FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
COPY apps/ui ./apps/ui

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[ui]"

EXPOSE 8501

CMD ["streamlit", "run", "apps/ui/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
