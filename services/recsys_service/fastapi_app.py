"""
FastAPI-приложение для сервиса рекомендаций банковских продуктов.

Основные обрабатываемые запросы:
- /pop_recs - получение рекомендаций по умолчанию
- /user_recs - получение рекомендаций для заданного user_id

Для запуска перейти на терминале в папку services/ и выполнить команду:
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

import logging
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np


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

# Создаем логгер
logger = logging.getLogger("uvicorn.error")


class Recommendations:
    """
    Класс для обработки запросов
    """
    def __init__(self):
        self._recs = {"june_2016": None, "default": None}
        
        # Метрики для мониторинга
        self._stats = {
            "existing_clients_with_recs_requests_count": 0,
            "existing_clients_wo_recs_requests_count": 0,
            "new_clients_requests_count": 0
        }

    def load(self, type, path, **kwargs):
        """
        Загружает рекомендации из файла
        """
        logger.info(f"Loading recommendations, type: {type}")
        
        if type == "june_2016":
            self._recs[type] = pd.read_csv(path, **kwargs)    
        elif type == "default":
            self._recs[type] = pd.read_parquet(path, **kwargs)

    def get_pop_recs(self, top_k : int = 7):
        """
        Возвращает список рекомендаций по умолчанию
        """
        pop_ranked_prods = self._recs["default"] 
        recs = list(pop_ranked_prods['eng_name'][:top_k])
        self._stats["new_clients_requests_count"] += 1
        
        return recs
    
    def get_user_recs(self, user_id: int, top_k: int = 7):
        """
        Возвращает список рекомендаций для заданного клиента
        """
        june_2016_recs = self._recs["june_2016"] 
        pop_ranked_prods = self._recs["default"] 
        
        recs = []
        try:
            user_df = june_2016_recs[june_2016_recs['ncodpers'] == user_id]
            user_prods = user_df[topn_pop_prods_added_pred_cols].to_numpy()[0]

            if user_prods.sum() != 0:
                # У клиента есть ненулевые рекомендации
                recommended_prods_idxs = np.argwhere(user_prods)[0]
                recs = list(pop_ranked_prods['eng_name'][recommended_prods_idxs])[:7]
            else:
                # У клиента нет рекомендаций (группа "не беспокоить")
                return []
        
        except IndexError:
            # Клиент не найден, предлагаем популярные продукты по умолчанию
            recs = list(pop_ranked_prods['eng_name'][:7])

        return recs

    def stats(self):
        logger.info("Requests statistics")
        for name, value in self._stats.items():
            logger.info(f"{name:<30} {value} ")


# Создаем объект для работы с рекомендациями
recs_store = Recommendations()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # код ниже (до yield) выполнится только один раз при запуске сервиса
    logger.info("Starting")

    # Загружаем рекомендации из файлов
    
    logger.info(f"Loading June 2016 recommendations")
    recs_store.load('june_2016', "./recs_data/june_2016_recs.csv.zip")
    logger.info(f"Loaded")

    logger.info(f"Loading popular products")
    recs_store.load('default', "./recs_data/pop_ranked_products.parquet")
    logger.info(f"Loaded")
    
    yield
    # этот код выполнится только один раз при остановке сервиса
    recs_store.stats()
    logger.info("Stopping")
    

# создаём приложение FastAPI
app = FastAPI(title="Bank Products Recommender System", lifespan=lifespan)


# Обращение к корневому url для проверки работоспособности сервиса
@app.get("/")
def read_root():
    return {"message": "Bank Products Recommender Service is working"}


# Получение рекомендаций по умолчанию из числа популярных продуктов
@app.post("/pop_recs")
async def recommendations_default(top_k: int = 7):
    """
    Возвращает список рекомендаций по умолчанию длиной top_k
    """
    recs = recs_store.get_pop_recs(top_k)
    return {"recs": recs}


# Получение готовых персональных рекомендаций
@app.post("/user_recs")
async def recommendations_offline(user_id: int = 1351337, top_k: int = 7):
    """
    Возвращает список оффлайн-рекомендаций длиной top_k для пользователя user_id
    """
    recs = recs_store.get_user_recs(user_id, top_k)
    return {"recs": recs}


if __name__ == "__main__":
    uvicorn.run("recsys_app:app", host="0.0.0.0", port="1702")