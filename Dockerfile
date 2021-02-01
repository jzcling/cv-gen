FROM python:3.7

COPY ./requirements/requirements.txt ./requirements/requirements.txt
RUN pip3 install -r requirements/requirements.txt

COPY ./app /app
RUN useradd -m container
USER container

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]