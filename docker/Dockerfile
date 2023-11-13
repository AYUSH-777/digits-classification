FROM ubuntu:23.10
FROM python:3.9.17
COPY . /digits/

RUN pip3 install -r /digits/requirements.txt

WORKDIR /digits
#CMD ["pytest"]

ENV FLASK_APP=api/hello

#CMD ["flask","run"]
CMD ["flask","run","--host=0.0.0git status.0"]