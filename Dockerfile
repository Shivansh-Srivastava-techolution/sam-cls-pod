FROM tensorflow/tensorflow:2.13.0-gpu

RUN apt-get update || true

COPY requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip
RUN apt-get install gcc -y
RUN pip3 install -r requirements.txt

COPY . /app
WORKDIR /app
EXPOSE 8501

ENTRYPOINT [ "python3" ]
CMD [ "server.py" ]
