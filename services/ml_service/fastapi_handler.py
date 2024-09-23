"""Класс FastApiHandler для обработки запросов к сервису рекомендации банковских продуктов.

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
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer


class FastApiHandler:
    """
    Класс FastApiHandler для обработки запросов к FastAPI-сервису 
    рекомендации банковских продуктов.
    """

    def __init__(self, 
                 test_data_path="./data/clean_data_test.csv.zip",
                 pop_ranked_prods_path="./data/pop_ranked_prods.parquet",
                 data_preprocessor_path="./model/data_preprocessor_fs.pkl",
                 model_path="./models/cb_clf_final.cbm"):
        """Метод для инициализации переменных класса."""

        # Типы параметров запроса для проверки
        self.param_types = {
            "model_params": dict
        }

        # Словарь со всеми обязательными параметрами модели и допустимыми типами их значений
        self.required_model_params = {
            'ncodpers':[int],
            'top_k':[int]
        }

        # Описание ошибки
        self.err_msg = ''

        # Загружаем тестовые данные, откуда будем брать параметры для модели
        self.load_test_data(test_data_path=test_data_path)

        # Загружаем данные о популярности продуктов для рекомендаций по умолчанию
        self.load_pop_ranked_prods(pop_ranked_prods_path=pop_ranked_prods_path)

        # Загружаем обученный трансформер данных
        self.load_data_preprocessor(data_preprocessor_path=data_preprocessor_path)

        # Загружаем обученный классификатор
        self.load_model(model_path=model_path)

    def load_test_data(self, data_path: str):
        try:
            self.test_data = pd.read_csv(data_path)
        except Exception as e:
            print(f"Failed to load test data, {e}")
            self.test_data = None

    def load_pop_ranked_prods(self, pop_ranked_prods_path: str):
        try:
            self.pop_ranked_prods = pd.read_parquet(pop_ranked_prods_path)
        except Exception as e:
            print(f"Failed to load products data, {e}")
            self.pop_ranked_prods = None

    def load_data_preprocessor(self, data_preprocessor_path: str):
        try:
            self.data_preprocessor = joblib.load(data_preprocessor_path)
        except Exception as e:
            print(f"Failed to load data preprocessor, {e}")
            self.data_preprocessor = None
        
    def load_model(self, model_path: str):
        try:
            self.cb_clf = CatBoostClassifier()
            self.cb_clf.load_model(model_path)
        except Exception as e:
            print(f"Failed to load model, {e}")
            self.clf = None

    def generate_recommendations(self, model_params: dict) -> float:
        """
        Метод для получения рекомендаций (параметры модели должны проверяться до вызова этого метода).
        Args:
        - model_params (dict): Параметры модели.
        Returns: список продуктов (float).
        """
        
        
        
        
        # Считаем возраст здания, т.к. наша модель ожидает этот параметр вместо года постройки
        model_params['building_age'] = datetime.now().year - model_params['build_year']
        # Удаляем лишний параметр
        del model_params['build_year']
        # Преобразуем в датафрейм
        model_params_df = pd.DataFrame(model_params, index=[0])        
        return self.pipeline.predict(model_params_df)[0]
        
    def check_required_query_params(self, query_params: dict) -> bool:
        """
        Метод для проверки параметров запроса.
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
        """
        Метод для проверки параметров модели.
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
        """
        Проверяем наличие и корректность всех параметров.
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
        """
        Функция для обработки FastAPI-запросов.
        Args:
        - params (dict): Параметры запроса.
        Returns:
        - Словарь с результатами выполнения запроса.
        """
        print('Processing request...')
        try:
            # Проверяем, были ли загружены тестовые данные
            if not self.test_data:
                response = {
                    'status': 'Error',
                    'message': "Test data not found"
                }
            # Проверяем, были ли данные о популярности продуктов
            elif not self.pop_ranked_prods:
                response = {
                    'status': 'Error',
                    'message': "Products data not found"
                }
            elif not self.data_preprocessor:
                response = {
                    'status': 'Error',
                    'message': "Data preprocessor not found"
                }
            # Проверяем, была ли загружена модель
            elif not self.clf:
                response = {
                    'status': 'Error',
                    'message': "Classifier not found"
                }
            # Валидируем запрос
            elif not self.validate_params(params):
                response = {
                    'status': 'Error',
                    'message': self.err_msg
                }
            else:
                model_params = params["model_params"]
                print("Generating recommendations...")
                recs = self.generate_recommendations(model_params)
                response = {
                    'status': 'OK',
                    'recs': recs 
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
            'ncodpers': 1000,
            'top_k': 7
        }
    }

    # Прописываем пути для общего случая, 
    # чтобы можно было запускать этот тест не только из папки services
    test_data_path = os.path.join(os.path.dirname(__file__), '../data/clean_data_test.csv.zip')
    pop_ranked_prods_path = os.path.join(os.path.dirname(__file__), '../data/pop_ranked_prods.parquet')
    data_preprocessor_path = os.path.join(os.path.dirname(__file__), '../models/data_preprocessor_fs.pkl')
    model_path = os.path.join(os.path.dirname(__file__), '../models/cb_clf_final.cbm')

    # Создаём обработчик запросов
    handler = FastApiHandler(test_data_path, 
                             pop_ranked_prods_path, 
                             data_preprocessor_path, 
                             model_path)

    # Обрабатываем тестовый запрос
    response = handler.handle(test_params)
    

if __name__ == "__main__":
    main()   
    
    