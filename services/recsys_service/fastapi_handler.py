"""
Класс FastApiHandler для обработки запросов к FastAPI-сервису рекомендаций банковских продуктов.

Чтобы протестировать этот файл без запуска uvicorn, нужно перейти в папку services/
и выполнить любую из двух команд: 

python -m recsys_service.fastapi_handler
или
python recsys_service/fastapi_handler.py
"""

import os
import numpy as np
import pandas as pd


# Имена колонок с предсказаниями покупок продуктов из числа top_n популярных
topn_pop_prods_added_pred_cols = [
    'ind_cco_fin_ult1_added_pred', 'ind_recibo_ult1_added_pred',
    'ind_ctop_fin_ult1_added_pred', 'ind_ecue_fin_ult1_added_pred',
    'ind_cno_fin_ult1_added_pred', 'ind_nom_pens_ult1_added_pred',
    'ind_nomina_ult1_added_pred', 'ind_reca_fin_ult1_added_pred',
    'ind_dela_fin_ult1_added_pred', 'ind_tjcr_fin_ult1_added_pred',
    'ind_ctpp_fin_ult1_added_pred', 'ind_valo_fin_ult1_added_pred',
    'ind_fond_fin_ult1_added_pred', 'ind_ctma_fin_ult1_added_pred',
    'ind_ctju_fin_ult1_added_pred', 'ind_plan_fin_ult1_added_pred',
    'ind_hip_fin_ult1_added_pred']


class FastApiHandler:
    """Класс FastApiHandler для обработки запросов к FastAPI-сервису рекомендаций банковских продуктов."""

    def __init__(self, 
                 pop_ranked_prods_path='./recs_data/pop_ranked_products.parquet', 
                 june_2016_recs_path='./recs_data/june_2016_recs.csv.zip',
                 logger=None):
        
        # Рекомендации
        self._recs = {"june_2016": None, "default": None}
        
        self.logger = logger

        # Загружаем рекомендации
        self.load('default', pop_ranked_prods_path)
        self.load('june_2016', june_2016_recs_path)
        

        # Метрики для мониторинга
        self._stats = {"existing_clients_with_recs_requests_count": 0,
                       "existing_clients_wo_recs_requests_count": 0,
                       "new_clients_requests_count": 0}

    def load(self, type, path, **kwargs):
        """
        Загружает рекомендации из файла
        """
        if self.logger:
            self.logger.info(f"Loading {type} recommendations..")
        
        if type == "june_2016":
            self._recs[type] = pd.read_csv(path, **kwargs)    
        elif type == "default":
            self._recs[type] = pd.read_parquet(path, **kwargs)
        
        if self.logger:
            self.logger.info(f"Loaded")

    def get_pop_recs(self, top_k : int = 7):
        """
        Возвращает список рекомендаций по умолчанию
        """
        pop_ranked_prods = self._recs["default"] 
        recs = list(pop_ranked_prods['eng_name'][:top_k])
        self._stats["new_clients_requests_count"] += 1
        
        response = {'status': 'OK',
                    'recs': recs}
        
        return response
    
    def get_user_recs(self, user_id: int, top_k: int = 7):
        """
        Возвращает список рекомендаций для заданного клиента
        """
        june_2016_recs = self._recs["june_2016"] 
        pop_ranked_prods = self._recs["default"] 
        
        try:
            user_df = june_2016_recs[june_2016_recs['ncodpers'] == user_id]
            user_prods = user_df[topn_pop_prods_added_pred_cols].to_numpy()[0]

            if user_prods.sum() != 0:
                # У клиента есть ненулевые рекомендации
                recommended_prods_idxs = np.argwhere(user_prods)[0]
                recs = list(pop_ranked_prods['eng_name'][recommended_prods_idxs])[:7]
                self._stats["existing_clients_with_recs_requests_count"] += 1
            else:
                # У клиента нет рекомендаций (группа "не беспокоить")
                recs = []
                self._stats["existing_clients_wo_recs_requests_count"] += 1
        
        except IndexError:
            # user_id не найден, считаем клиента новым и предлагаем ему популярные продукты
            recs = list(pop_ranked_prods['eng_name'][:7])
            self._stats["new_clients_requests_count"] += 1

        response = {'status': 'OK',
                    'recs': recs}
        
        return response

    def stats(self):
        """Записывает в логгер накопленную статистику"""
        if self.logger:
            self.logger.info("Requests statistics:")
            for name, value in self._stats.items():
                self.logger.info(f"{name:<30} {value} ")


def main():
    # Код для тестирования обработчика без запуска uvicorn
    
    # Прописываем пути, чтобы можно было запускать этот тест не только из папки services/
    pop_ranked_prods_path = os.path.join(os.path.dirname(__file__), '../recs_data/pop_ranked_products.parquet')
    june_2016_recs_path = os.path.join(os.path.dirname(__file__), '../recs_data/june_2016_recs.csv.zip')

    # Создаём обработчик запросов
    handler = FastApiHandler(pop_ranked_prods_path, june_2016_recs_path)

    # Тестовые примеры

    # Существующий клиент с user_id = 1351337, для которого есть рекомендации
    response = handler.get_user_recs(1351337, 7)
    print(response['recs'])

    # Существующий клиент с user_id = 418977, для которого нет рекомендаций (группа "не беспокоить")
    response = handler.get_user_recs(418977, 7)
    print(response['recs'])
    
    # Новый клиент без user_id
    response = handler.get_pop_recs()
    print(response['recs'])
    
    # Клиент с несуществующим user_id (обрабатываем его как нового)
    response = handler.get_user_recs(0, 7)
    print(response['recs'])
    
    

if __name__ == "__main__":
    main()   
      