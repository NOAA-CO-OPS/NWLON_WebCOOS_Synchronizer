FROM mambaorg/micromamba:2.0.2

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n nwlon_webcoos_synchronizer -f /tmp/environment.yml && \
    micromamba clean --all --yes
