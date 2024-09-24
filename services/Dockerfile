FROM python:3.11-slim

COPY . ./recs_app

WORKDIR recs_app

RUN pip3 install -r requirements_recs_service.txt

EXPOSE 1702

VOLUME /recs_data

CMD uvicorn recs_service.fastapi_app:app --reload --port 1702 --host  0.0.0.0
