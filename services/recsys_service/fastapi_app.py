"""
FastAPI-приложение для сервиса рекомендаций банковских продуктов.

Основные обрабатываемые запросы:
- /get_recs - получение рекомендаций

Для запуска перейти в папку services/recsys_service/ и выполнить команду:
uvicorn recsys_app:app --reload --port 8081 --host 0.0.0.0

либо, если работа ведется полностью локально:
uvicorn fastapi_app:app --reload --port 8081 --host 127.0.0.1

Если используется другой порт, то заменить 8081 на этот порт.

Для просмотра документации API и совершения тестовых запросов через 
Swagger UI зайти на  http://127.0.0.1:8081/docs

Для запуска и тестирования см. инструкции в файле README.md
"""

import logging
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np


# Создаем несколько вспомогательных структур

# Словарь для перевода названий продуктов на англ. яз.
prod2eng = {
    "ind_ahor_fin_ult1" : "Saving Account",
    "ind_aval_fin_ult1" : "Guarantees",
    "ind_cco_fin_ult1" : "Current Account",
    "ind_cder_fin_ult1" : "Derivada Account",
    "ind_cno_fin_ult1" : "Payroll Account",
    "ind_ctju_fin_ult1" : "Junior Account",
    "ind_ctma_fin_ult1" : "Más Particular Account",
    "ind_ctop_fin_ult1" : "Particular Account",
    "ind_ctpp_fin_ult1" : "Particular Plus Account",
    "ind_deco_fin_ult1" : "Short-term Deposits",
    "ind_deme_fin_ult1" : "Medium-term Deposits",
    "ind_dela_fin_ult1" : "Long-term Deposits",
    "ind_ecue_fin_ult1" : "E-account",
    "ind_fond_fin_ult1" : "Funds",
    "ind_hip_fin_ult1" : "Mortgage",
    "ind_plan_fin_ult1" : "Plan Pensions",
    "ind_pres_fin_ult1" : "Loans",
    "ind_reca_fin_ult1" : "Taxes",
    "ind_tjcr_fin_ult1" : "Credit Card",
    "ind_valo_fin_ult1" : "Securities",
    "ind_viv_fin_ult1" : "Home Account",
    "ind_nomina_ult1" : "Payroll",
    "ind_nom_pens_ult1" : "Pensions",
    "ind_recibo_ult1" : "Direct Debit"
}

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

    def get_default(self, top_k: int=7):
        """
        Возвращает список рекомендаций по умолчанию
        """
        pop_ranked_prods = self._recs["default"] 
        recs = list(pop_ranked_prods['eng_name'][:top_k])
        
        self._stats["new_clients_requests_count"] += 1
        logger.info(f"New clients requests count: {self._stats['new_clients_requests_count']}")

        return recs
    
    def get_user_recs(self, user_id: int, top_k: int = 7):
        """
        Возвращает список рекомендаций для заданного клиента
        """
        june_2016_recs = self._recs["june_2016"] 
        pop_ranked_prods = self._recs["default"] 
        
        recs = []
        try:
            topn_prods = june_2016_recs[june_2016_recs['ncodpers'] == user_id][topn_pop_prods_added_pred_cols]\
                .to_numpy()[0]
            try:
                rec_prods_idxs = np.argwhere(topn_prods)[0]
                # У клиента есть ненулевые рекомендации
                recs = list(pop_ranked_prods['eng_name'][rec_prods_idxs])[:top_k]

                self._stats["existing_clients_with_recs_requests_count"] += 1
                logger.info(f"Existing clients with recs requests count: \
                            {self._stats['existing_clients_with_recs_requests_count']}")
            except:
                # У клиента нет рекомендаций (группа "не беспокоить")
                self._stats["existing_clients_wo_recs_requests_count"] += 1
                logger.info(f"Existing clients without recs requests count: \
                            {self._stats['existing_clients_wo_recs_requests_count']}")
                return []
        
        except:
            # Клиент не найден, предлагаем популярные продукты по умолчанию
            recs = list(pop_ranked_prods['eng_name'][:top_k])

        return recs

    def stats(self):
        logger.info("Stats for recommendations")
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
@app.post("/get_recs")
async def recommendations_default(top_k: int = 7):
    """
    Возвращает список рекомендаций по умолчанию длиной top_k
    """
    recs = recs_store.get_default(top_k)
    return {"recs": recs}


# Получение готовых персональных рекомендаций
@app.post("/get_recs")
async def recommendations_offline(user_id: int, top_k: int = 7):
    """
    Возвращает список оффлайн-рекомендаций длиной top_k для пользователя user_id
    """
    recs = recs_store.get_user_recs(user_id, top_k)
    return {"recs": recs}


if __name__ == "__main__":
    uvicorn.run("recsys_app:app", host="0.0.0.0", port="8081")