FROM heurekalabs/syft-base

RUN apt-get update && \
    apt-get install -y gnuplot && \
    apt-get clean

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt

COPY . /app
WORKDIR /app
RUN rm -f /app/**/*.db

ENTRYPOINT ["sh", "entrypoint.sh"]
