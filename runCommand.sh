sudo docker build -t asher-training-pod .
sudo docker run -d --restart=always -p 8501:8501 --gpus all --name asher-training-pod asher-training-pod
