FROM python:3.8-slim AS base
WORKDIR /app
COPY requirements.txt .

# Upgrade pip
RUN python -m pip install --upgrade pip
