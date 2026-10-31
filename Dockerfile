FROM mambaorg/micromamba:1.5.10

# 1) Dependências
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

# 2) Dados (camada pesada, muda raramente)
COPY --chown=$MAMBA_USER:$MAMBA_USER data /data
ENV LEPTO_DATA_DIR=/data

# 3) Código (muda com frequência)
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER main.py html_pages.py geradorPCA.py ./

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]