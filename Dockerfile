# Docker alternative to `mamba env create -f bioenv.yml`.
# The image contains only the software environment; the repository itself is
# mounted at /work when the container is started (see README.md).
#
# bioenv.yml pins linux-64 builds, so the image is linux/amd64 only.
FROM --platform=linux/amd64 mambaorg/micromamba:2.3.2

# Install the pinned environment into the image's default (auto-activated) env.
COPY --chown=$MAMBA_USER:$MAMBA_USER bioenv.yml /tmp/bioenv.yml
RUN micromamba install -y -n base -f /tmp/bioenv.yml \
    && micromamba clean --all --yes \
    && rm /tmp/bioenv.yml

# Lets the container run as the host user (docker run --user), so files written
# to the mounted repository are owned by that user.
USER root
RUN mkdir -p /home/bioenv /work && chmod 1777 /home/bioenv /work
USER $MAMBA_USER

ENV HOME=/home/bioenv \
    PATH=/work/cg_sims/scripts:$PATH

WORKDIR /work
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
