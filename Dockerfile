# hetudit — thin app image, layered on top of hetudit-base.
#
# Holds only the hetu_dit source tree and its editable install. All third-party
# deps (pytorch, ray, transformers, flash-attn, ...) come from the base image.
# Rebuild on every hetu_dit/ source change; see k8s/IMAGE_LAYERS.md.

ARG BASE_TAG=latest
FROM hetudit-base:${BASE_TAG}

WORKDIR /workspace/Hetu-DiT

COPY --chown=10001:10001 . /workspace/Hetu-DiT

# --no-deps: every dep already lives in the base image; if pip thinks
# something is missing here, fix it by adding the dep to requirements-base.txt
# and rebuilding the base image — never let it sneak into the app layer.
# --no-build-isolation: the base image already has setuptools; without this
# flag pip would spin up a clean venv and re-fetch setuptools from pypi,
# which both wastes time and requires network egress for every app build.
RUN python3 -m pip install -e . --no-deps --no-build-isolation && \
    mkdir -p /tmp/ray && \
    chown -R 10001:10001 /tmp/ray

USER 10001:10001

EXPOSE 8000

CMD ["python3", "-m", "hetu_dit.entrypoint.api_server", "--host", "0.0.0.0", "--port", "8000"]
