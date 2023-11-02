docker build -t digits:v1 -f docker/Dockerfile .

docker run -v ./models:/digits/models digits:v1