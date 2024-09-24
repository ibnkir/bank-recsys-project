FROM python:3.11-slim

COPY . ./recsys_app

WORKDIR recsys_app

RUN pip3 install -r requirements_recsys_service.txt

EXPOSE 1702

VOLUME /recs_data

CMD uvicorn recsys_service.fastapi_app:app --reload --port 1702 --host  0.0.0.0
