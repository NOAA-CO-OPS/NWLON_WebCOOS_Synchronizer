FROM mambaorg/micromamba:2.0.2

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n Data_Webcam_Synchronizer -f /tmp/environment.yml && \
    micromamba clean --all --yes
