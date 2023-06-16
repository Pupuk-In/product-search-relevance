FROM python:3.9

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

WORKDIR /app

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]