FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user && \
    mkdir -p /app /home/user/.cache && \
    chown -R user:user /app /home/user

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "app.py"]
