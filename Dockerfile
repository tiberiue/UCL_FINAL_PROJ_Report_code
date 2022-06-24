FROM atlas/centos7-atlasos

# Create the environment:
COPY env1.yml .
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh && \
    bash Miniconda3-py39_4.9.2-Linux-x86_64.sh -b -p /miniconda
ADD ./ /ljptagger

#RUN bash Miniconda3-py39_4.9.2-Linux-x86_64.sh -b -p /miniconda
RUN source /miniconda/bin/activate && \
    conda env create -f env1.yml
