FROM ubuntu:23.10
FROM python:3.9.17
COPY . /digits/

RUN pip3 install -r /digits/requirements.txt

WORKDIR /digits

VOLUME /digits/models

CMD ["pytest"]

CMD python /digits/digits_classification.py
