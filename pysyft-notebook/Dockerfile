FROM heurekalabs/syft-base

ENV WORKSPACE /workspace
ENV CODE /code
ENV NOTEBOOKS /notebooks

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# Setup workspace environment
RUN pip3 install --no-cache-dir \
        syft[udacity,sandbox,notebooks] \
        pandas \
        seaborn \
        notebook \
        jupyterlab

# checkout github repository with notebooks
RUN mkdir $CODE
RUN git clone https://github.com/matthiaslau/Turbofan-Federated-Learning-POC.git $CODE
RUN mv $CODE/notebooks $NOTEBOOKS

# Create jupyter notebook workspace
RUN mkdir $WORKSPACE
WORKDIR $WORKSPACE

# Make the image start the jupyer notebook
COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
