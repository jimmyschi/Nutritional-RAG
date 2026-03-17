FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /mlflow

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir mlflow==2.17.2

EXPOSE 5000
