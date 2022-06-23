FROM continuumio/miniconda3

ADD /eos/user/j/jzahredd/JetTagging/miniconda3 /miniconda3
ADD /afs/cern.ch/work/j/jzahredd/LJPTagger/gitlab/ljptagger /ljptagger

RUN source /miniconda3/bin/activate && \
    conda activate rootenv

WORKDIR /ljptagger
