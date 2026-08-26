# Hugging Face Spaces deploys Streamlit apps via the Docker SDK now --
# Streamlit is no longer a standalone Spaces SDK option. This follows HF's
# documented Docker Spaces conventions: non-root user 1000, app under
# $HOME/app, port 7860.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

RUN pip install --no-cache-dir --upgrade pip

# Only the runtime requirements -- requirements-dev.txt (reportlab,
# sentence-transformers, torch) is for local demo/test tooling only and is
# never needed by the deployed app itself.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
