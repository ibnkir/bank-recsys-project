"""
FastAPI-приложение для сервиса рекомендаций банковских продуктов.

Основные обрабатываемые запросы:
- /pop_recs - получение рекомендаций по умолчанию
- /user_recs - получение рекомендаций для заданного user_id

Для запуска без докера перейти на терминале в папку services/ и выполнить команду:
uvicorn recsys_service.fastapi_app:app --reload --port 1702 --host 0.0.0.0

либо, если работа ведется полностью локально:
uvicorn recsys_service.fastapi_app:app --reload --port 1702 --host 127.0.0.1

Убедиться, что сервис поднялся, можно перейдя по ссылке:
http://localhost:1702/

Отправка тестовых запросов:
- Swagger UI: http://localhost:1702/docs
- С помощью скрипта test_services.py (см. инструкции в readme.md)

Для остановки uvicorn использовать Ctrl+C
"""
from fastapi import FastAPI
from .fastapi_handler import FastApiHandler
import logging
import uvicorn
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np


# Создаем логгер
logger = logging.getLogger("uvicorn.error")


# Функция, которая выполняется в начале и конце работы сервиса
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код до yield выполнится только один раз при запуске сервиса
    logger.info("Starting")

    # Создаем обработчик запросов
    app.handler = FastApiHandler(pop_ranked_prods_path='./recs_data/pop_ranked_products.parquet', 
                                 june_2016_recs_path='./recs_data/june_2016_recs.csv.zip',
                                 logger=logger)

    yield
    # Этот код выполнится только один раз при остановке сервиса
    # Выводим в логгер статистику
    app.handler.stats()
    logger.info("Stopping")
    

# создаём приложение FastAPI
app = FastAPI(title="Bank Products Recommender System", lifespan=lifespan)


# Обращение к корневому url для проверки работоспособности сервиса
@app.get("/")
async def read_root():
    return {"message": "Bank Products Recommender Service is working"}


# Получение рекомендаций по умолчанию из числа популярных продуктов
@app.post("/pop_recs")
async def recommendations_default(top_k: int = 7):
    """Возвращает словарь с полем recs, которое содержит список рекомендаций по умолчанию длиной top_k"""
    response = app.handler.get_pop_recs(top_k)
    return response


# Получение готовых персональных рекомендаций
@app.post("/user_recs")
async def recommendations_offline(user_id: int = 1351337, top_k: int = 7):
    """
    Возвращает словарь с полем recs, которое содержит список оффлайн-рекомендаций длиной top_k для пользователя user_id
    """
    response = app.handler.get_user_recs(user_id, top_k)
    return response


if __name__ == "__main__":
    uvicorn.run("recsys_app:app", host="0.0.0.0", port="1702")