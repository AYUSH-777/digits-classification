FROM ubuntu:23.10
FROM python:3.9.17
COPY . /digits/

RUN pip3 install -r /digits/requirements.txt

WORKDIR /digits
CMD ["pytest"]
