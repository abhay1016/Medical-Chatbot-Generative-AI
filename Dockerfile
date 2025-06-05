From python:3.10-slim-buster
WORKDIR /app
COPY . /app
run pip install -r requirements.txt
CMD ["python3","app.py"]