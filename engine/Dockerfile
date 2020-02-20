FROM heurekalabs/syft-base

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt

COPY . /app
WORKDIR /app

ENTRYPOINT ["sh", "entrypoint.sh"]
