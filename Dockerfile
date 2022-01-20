FROM python:3.8-slim

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
ADD . /code
WORKDIR /code

ENTRYPOINT ["python", "read_data.py"]

 