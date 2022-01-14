FROM python:3.8

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
ADD . /code
WORKDIR /code

ENTRYPOINT ["python", "read_data.py"]

 