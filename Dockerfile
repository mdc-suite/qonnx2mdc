FROM ubuntu:20.04

# In order to avoid interactive questions
ARG DEBIAN_FRONTEND=noninteractive

RUN mkdir /opt/python3.10.10

RUN apt update
RUN apt install -qq -y wget libffi-dev gcc build-essential curl tcl-dev tk-dev uuid-dev lzma-dev liblzma-dev libssl-dev libsqlite3-dev git libbz2-dev

RUN wget https://www.python.org/ftp/python/3.10.10/Python-3.10.10.tgz
RUN tar -zxvf Python-3.10.10.tgz
RUN cd Python-3.10.10 && ./configure --prefix=/opt/python3.10.10 && make && make install

RUN rm Python-3.10.10.tgz && rm -r Python-3.10.10/

RUN ln -s /opt/python3.10.10/bin/python3.10 /usr/bin/python3
RUN ln -s /opt/python3.10.10/bin/pip3.10 /usr/bin/pip3

COPY requirements.txt .
RUN pip3 install -r requirements.txt

#In order to be able to run notebok (Jupyter Notebook)
EXPOSE 8888
