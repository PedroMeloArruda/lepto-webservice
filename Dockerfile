FROM mambaorg/micromamba:1.5.10

# 1) Instala as dependências (esta camada só refaz se o environment.yml mudar)
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

# 2) Copia o código
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER main.py html_pages.py geradorPCA.py ./

# 3) Os dados ficam fora da imagem, montados em /data
ENV LEPTO_DATA_DIR=/data

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]