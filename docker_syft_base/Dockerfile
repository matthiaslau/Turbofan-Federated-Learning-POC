FROM python:3.7-slim

RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get clean && \
    pip3 install \
        msgpack==0.6.2 \
        numpy \
        pandas \
        sklearn \
        syft==0.2.2a1 \
        torch==1.3.0 \
