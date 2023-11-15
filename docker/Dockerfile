FROM ubuntu:23.10
FROM python:3.9.17
COPY . /digits/
WORKDIR /digits

RUN pip3 install -r /digits/requirements.txt


ENV FLASK_APP=api/hello

#CMD ["flask","run"]

CMD ["flask","run","--host=0.0.0.0"]

#VOLUME /digits/models
#CMD ["pytest"]
#CMD python /digits/digits_classification.py