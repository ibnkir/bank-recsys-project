"""FastAPI-приложение для предсказания цен на квартиры по заданным параметрам.

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
        router.add_api_route("/predict", self.get_prediction_for_item, methods=["POST"])
        self.app.include_router(router)

    def get_app(self):
        return self.app
    
    def read_root(self):
        return {'message': 'Welcome from the FastAPI'}
    
    def get_prediction_for_item(
        self,
        model_params: dict = Body(
            example={
                'floor': 6,
                'kitchen_area': 8.5,
                'living_area': 30.0,
                'rooms': 2,
                'is_apartment': False,
                'total_area': 50.0,
                'build_year': 1979,
                'building_type_int': 4,
                'latitude': 60.0,
                'longitude': 40.0,
                'ceiling_height': 2.5,
                'flats_count': 190,
                'floors_total': 12,
                'has_elevator': True
            } 
        )                                                   
    ):
        """Функция для получения и обработки запроса к предсказательному сервису ml_service.<br>
        Args:<br>
            - params (dict): Параметры запроса.<br>
        Returns: ответ в формате JSON с предсказанием цены в поле 'score' 
        либо описанием ошибки в поле 'message'.
        """
        all_params = {
            "model_params": model_params
        }  
        
        print('Processing request...')
        response = self.app.handler.handle(all_params)
        
        if 'status' in response and response['status'] == 'OK':
            self.ml_service_predictions.observe(response['score'])
        else:
            self.ml_service_err_requests.inc()

        return response


app = FastAPIWrapper().get_app()


"""
# Альтернативный вариант без оборачивания в класс

# Создаём приложение FastAPI
app = FastAPI()

# Создаём обработчик запросов для API
app.handler = FastApiHandler()

# Инициализируем и запускаем экпортёр метрик
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Метрика-гистограмма с предсказаниями модели
ml_service_predictions = Histogram(
    "ml_service_predictions",
    "Histogram of predictions",
    buckets=(0.8e7, 0.9e7, 1.0e7, 1.5e7)
)

# Метрика-счетчик запросов с неправильными параметрами
ml_service_err_requests = Counter(
    "ml_service_err_requests", 
    "Counter of requests with wrong parameters"
)

@app.get("/")
def read_root():
    return {'message': 'Welcome from the FastAPI'}

@app.post("/predict") 
def get_prediction_for_item(
    model_params: dict = Body(
        example={
            'floor': 6,
            'kitchen_area': 8.5,
            'living_area': 30.0,
            'rooms': 2,
            'is_apartment': False,
            'total_area': 50.0,
            'build_year': 1979,
            'building_type_int': 4,
            'latitude': 60.0,
            'longitude': 40.0,
            'ceiling_height': 2.5,
            'flats_count': 190,
            'floors_total': 12,
            'has_elevator': True
        } 
    )                                                   
):
    all_params = {
        "model_params": model_params
    }  
    
    print('Processing request...')
    response = app.handler.handle(all_params)
    
    if 'status' in response and response['status'] == 'OK':
        ml_service_predictions.observe(response['score'])
    else:
        ml_service_err_requests.inc()

    return response
"""
