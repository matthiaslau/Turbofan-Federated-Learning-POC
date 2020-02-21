FROM heurekalabs/syft-base

ENV WORKSPACE /workspace

# Setup workspace environment
RUN pip3 install --no-cache-dir \
        syft[udacity,sandbox,notebooks] \
        pandas \
        seaborn \
        notebook \
        jupyterlab

# Create jupyter notebook workspace
RUN mkdir $WORKSPACE
WORKDIR $WORKSPACE

# Make the image start the jupyer notebook
COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
