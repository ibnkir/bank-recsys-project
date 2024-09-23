"""FastAPI-приложение для рекомендации банковских продуктов.

Для запуска без докера перейти в папку services/ и выполнить команду:
uvicorn ml_service.fastapi_app:app --reload --port 1702 --host 0.0.0.0

либо, если работа ведется локально:
uvicorn ml_service.fastapi_app:app --reload --port 1702 --host 127.0.0.1

Для просмотра документации API и совершения тестовых запросов через 
Swagger UI перейти в браузере по ссылке  http://127.0.0.1:1702/docs
Для отправки простого get-запроса можно ввести в терминале команду
curl http://127.0.0.1:1702/
"""

from fastapi import FastAPI, Body, APIRouter
from .fastapi_handler import FastApiHandler
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import (
    Histogram,
    Counter
)


# Оборачиваем FastAPI-приложение в класс
class FastAPIWrapper:
    def __init__(self):
        
        # Создаём приложение FastAPI
        self.app = FastAPI()

        # Создаём обработчик запросов
        self.app.handler = FastApiHandler()

        # Инициализируем и запускаем экпортёр метрик
        self.instrumentator = Instrumentator()
        self.instrumentator.instrument(self.app).expose(self.app)

        # Метрика-гистограмма с предсказаниями модели
        self.ml_service_predictions = Histogram(
            "ml_service_predictions",
            "Histogram of predictions",
            buckets=(0.8e7, 0.9e7, 1.0e7, 1.5e7)
        )

        # Метрика-счетчик запросов с неправильными параметрами
        self.ml_service_err_requests = Counter(
            "ml_service_err_requests", 
            "Counter of requests with wrong parameters"
        )

        router = APIRouter()
        router.add_api_route("/", self.read_root, methods=["GET"])
        router.add_api_route("/get_recs", self.get_recs_for_client, methods=["POST"])
        self.app.include_router(router)

    def get_app(self):
        return self.app
    
    def read_root(self):
        return {'message': 'Welcome to the Bank Products Recommender FastAPI-service'}
    
    def get_recs_for_client(
        self,
        model_params: dict = Body(
            example={
                'ncodpers': 1000,
                'top_k': 7
            } 
        )                                                   
    ):
        """
        Функция для получения и обработки запроса к сервису ml_service.<br>
        Args:<br>
        - model_params (dict): Параметры модели, включая ID клиента 
        и кол-во показываемых рекомендаций.<br>
        Returns: ответ в формате JSON с рекомендованными продуктами в поле 'recs' 
        либо описанием ошибки в поле 'message'.
        """
        all_params = {
            "model_params": model_params
        }  
        
        print('Processing request...')
        response = self.app.handler.handle(all_params)
        
        if 'status' in response and response['status'] == 'OK':
            self.ml_service_predictions.observe(response['recs'])
        else:
            self.ml_service_err_requests.inc()

        return response


app = FastAPIWrapper().get_app()


