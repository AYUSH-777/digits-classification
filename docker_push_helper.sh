Steps alongwith the assosciated commands are mentioned as below:

1. building and running the depndency docker image i.e DockerfileDepndecy :

docker build -t digits:v1 -f  DockerfileDependency.

docker run -it digits:v1


2. building and running the final docker image i.e DockerfileFinal :

docker build -t digits:v1 -f DockerfileFinal .

docker run -it digits:v1


3. login inside the azure via terminal


az login --use-device

4. pushing the dependency image as base on azure

az acr build --registry ayushmlops23 --image base --file DockerfileDependency .

5. pushing the final image as digits on azure

az acr build --registry ayushmlops23 --image digits --file DockerfileFinal .




conda create -m digits python=3.9

conda activate digits

pip install -r requirements.txt

python app.py
