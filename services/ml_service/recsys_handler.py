"""Класс FastApiHandler для обработки запросов к FastAPI-сервису предсказания цен на квартиры.

Чтобы протестировать этот файл без запуска uvicorn, нужно перейти в папку services/
и выполнить любую из двух команд: 

python -m ml_service.fastapi_handler
или
python ml_service/fastapi_handler.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime


class FastApiHandler:
    """Класс FastApiHandler для обработки запросов к FastAPI-сервису предсказания цен на квартиры."""

    def __init__(self, model_path="./models/flats_prices_fitted_pipeline.pkl"):
        """Метод для инициализации переменных класса."""

        # Типы параметров запроса для проверки
        self.param_types = {
            "model_params": dict
        }

        # Словарь со всеми обязательными параметрами модели и допустимыми типами их значений
        self.required_model_params = {
            'floor':[int], 
            'kitchen_area':[float, int], 
            'living_area': [float, int], 
            'rooms': [int], 
            'is_apartment': [bool, int],
            'total_area': [float, int], 
            'build_year': [int], 
            'building_type_int': [int], 
            'latitude': [float, int], 
            'longitude': [float, int], 
            'ceiling_height': [float, int],
            'flats_count': [int], 
            'floors_total': [int], 
            'has_elevator': [bool, int]
        }

        # Описание ошибки
        self.err_msg = ''

        # Загружаем обученную ценовую модель
        self.load_price_model(model_path=model_path)
        
    def load_price_model(self, model_path: str):
        """Метод для загрузки обученной ценовой модели.
        Args:
            - model_path (str): Путь до модели.
        """
        try:
            self.pipeline = joblib.load(model_path)
        except Exception as e:
            print(f"Failed to load model, {e}")
            self.pipeline = None

    def price_predict(self, model_params: dict) -> float:
        """Метод для получения прогнозной цены (параметры модели должны проверяться до вызова этого метода).
        Args:
            - model_params (dict): Параметры модели.
        Returns: Прогнозная цена (float).
        """
        # Считаем возраст здания, т.к. наша модель ожидает этот параметр вместо года постройки
        model_params['building_age'] = datetime.now().year - model_params['build_year']
        # Удаляем лишний параметр
        del model_params['build_year']
        # Преобразуем в датафрейм
        model_params_df = pd.DataFrame(model_params, index=[0])        
        return self.pipeline.predict(model_params_df)[0]
        
    def check_required_query_params(self, query_params: dict) -> bool:
        """Метод для проверки параметров запроса.
        Args:
            - query_params (dict): Параметры запроса.
        Returns: True, если есть нужные параметры, иначе False .
        """
        if 'model_params' not in query_params \
            or not isinstance(query_params["model_params"], self.param_types['model_params']):
            self.err_msg = "Not all query params exist"
            print(self.err_msg)
            return False
                
        return True
    
    def check_required_model_params(self, model_params: dict) -> bool:
        """Метод для проверки параметров модели.
        Args:
            - model_params (dict): Параметры модели.
        Returns: True, если есть все требуемые параметры, их типы соответствуют заданным и 
        выполнены предусмотренные ограничения, иначе False.
        """
        # Этот признак мы не использовали при обучении модели
        if 'studio' in model_params:
            del model_params['studio']
        
        # Проверяем наличие всех требуемых параметров модели
        if model_params.keys() != self.required_model_params.keys():
            self.err_msg = "There are missing or extra model params"
            print(self.err_msg)
            return False
               
        
        # Проверяем, что типы значений соответствуют заданным и все числовые параметры положительны
        for k, v in model_params.items():
            # Проверяем типы значений
            if not type(v) in self.required_model_params[k]:
                self.err_msg = 'Some features in model params have wrong value type'
                print(self.err_msg)
                return False
            # Проверяем булевские параметры, если они переданы как int
            elif type(v) == int and k in {'is_apartment', 'has_elevator'} and v not in {0, 1}:
                self.err_msg = 'Features is_apartment and has_elevator should be boolean or 0/1'
                print(self.err_msg)
                return False
            elif type(v) in {float, int} and k != 'building_type_int' and v < 0:
                self.err_msg = 'Some numerical features in model params are negative'
                print(self.err_msg)
                return False
            
        # Проверяем, что год постройки не превышает текущий год
        if model_params['build_year'] > datetime.now().year or \
            model_params['build_year'] < 1900:
            self.err_msg = "Parameter build_year should be between 1900 and current year"
            print(self.err_msg)
            return False
        
        # Проверяем, что тип здания лежит в диапазоне [0..6]
        # (в обучающей выборке были только такие значения)
        if model_params['building_type_int'] < 0 or model_params['building_type_int'] > 6:
            self.err_msg = "Parameter building_type_int should be in the range [0..6]"
            print(self.err_msg)
            return False

        return True
    
    def validate_params(self, params: dict) -> bool:
        """Проверяем наличие и корректность всех параметров.
        Args:
            - params (dict): Параметры запроса.
        Returns: True, если все параметры корректны, иначе False.
        """
        # Проверяем параметры запроса
        if self.check_required_query_params(params):
            print("All query params exist")
        else:
            return False
        
        # Проверяем параметры модели
        if self.check_required_model_params(params['model_params']):
            print("All model params exist and correct")
            return True
        
        return False
		
    def handle(self, params):
        """Функция для обработки FastAPI-запросов.
        Args:
            - params (dict): Параметры запроса.
        Returns:
            - Словарь с результатами выполнения запроса.
        """
        print('Processing request...')
        try:
            # Проверяем, была ли загружена модель
            if not self.pipeline:
                response = {
                    'status': 'Error',
                    'message': "Model not found"
                }
            # Валидируем запрос
            elif not self.validate_params(params):
                response = {
                    'status': 'Error',
                    'message': self.err_msg
                }
            else:
                model_params = params["model_params"]
                print("Making prediction...")
                y_pred = self.price_predict(model_params)
                response = {
                    'status': 'OK',
                    'score': y_pred 
                }    
        
        except Exception as e:
            response = {
                'status': 'Error',
                'message': f'Problem with request, {e}'
            }
            print(response)
            return response
        
        else:
            print(response)
            return response


def main():
    # Создаём параметры для тестового запроса
    test_params = {
        'model_params': {
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
    }

    # Прописываем путь для общего случая, 
    # чтобы можно было запускать этот тест не только из папки services
    model_path = os.path.join(os.path.dirname(__file__), '../models/flats_prices_fitted_pipeline.pkl')

    # Создаём обработчик запросов
    handler = FastApiHandler(model_path)

    # Обрабатываем тестовый запрос
    response = handler.handle(test_params)
    

if __name__ == "__main__":
    main()   
    
    