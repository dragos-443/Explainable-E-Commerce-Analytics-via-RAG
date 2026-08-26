ARG SPARK_IMAGE=apache/spark:3.5.5
FROM ${SPARK_IMAGE}

USER root

COPY requirements-container.txt /tmp/requirements-container.txt
RUN pip install --no-cache-dir -r /tmp/requirements-container.txt
COPY docker/app/sitecustomize.py /usr/lib/python3.8/sitecustomize.py
RUN mkdir -p /tmp/app-home && chown -R 185:0 /tmp/app-home

WORKDIR /workspace

USER 185
