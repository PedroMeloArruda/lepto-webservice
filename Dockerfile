FROM mambaorg/micromamba:1.5.10

#dependecias
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

# dados
COPY --chown=$MAMBA_USER:$MAMBA_USER data /data
ENV LEPTO_DATA_DIR=/data

# codigo e segurança
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER main.py html_pages.py geradorPCA.py ./

USER $MAMBA_USER
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD micromamba run -n base python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/', timeout=3)" || exit 1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
