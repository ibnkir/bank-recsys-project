"""
Вспомогательный скрипт для отправки тестовых запросов из командной строки.

Основные реализованные функции:
- get_pop_recs - получение рекомендаций по умолчанию из числа топ-популярных продуктов;
- get_user_recs - получение персональных рекомендаций по user_id.

Для запуска и тестирования см. инструкции в файле README.md
"""

import numpy as np
import pandas as pd
import requests
import json
import logging
import sys
import argparse
import configparser


# Настраиваем логирование
logger = logging.getLogger('test_logs')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s, %(funcName)s, %(message)s',
                    handlers=[logging.FileHandler("test_service.log", mode='a'),
                              stream_handler])


# Создаем парсер конфигурационного файла
config = configparser.ConfigParser()
config.read("config.ini")  

# Читаем url-адрес сервиса из конфигурационного файла "http://127.0.0.1:1702"
recsys_url = config["urls"]["recsys_url"] 

# Общий заголовок для всех http-запросов
headers = {"Content-type": "application/json", "Accept": "text/plain"}


# Получение рекомендаций по умолчанию из числа топ-популярных продуктов
def get_pop_recs(top_k: int = 7):
    """
    Возвращает названия top_k популярных продуктов
    """
    params = {'top_k': top_k}
    resp = requests.post(recsys_url + "/get_pop_recs", headers=headers, params=params)
    if resp.status_code == 200:
        resp = resp.json()
        logger.info(f"top-{top_k} popular products: {resp['recs']}")
    else:
        logger.info(f"Error, status code: {resp.status_code}")
    

# Получение персональных рекомендаций по user_id
def get_user_recs(user_id: int = 617032, top_k: int = 7):
    params = {'user_id': user_id, 'top_k': top_k}
    resp = requests.post(recsys_url + "/get_user_recs", headers=headers, params=params)
    if resp.status_code == 200:
        resp = resp.json()
        logger.info(f"Recommendations for user_id={user_id}: {resp['recs']}")
    else:
        logger.info(f"Error, status code: {resp.status_code}")
    
    return resp
    

if __name__ == "__main__":
  
    # Создаем парсер для чтения аргументов, передаваемых из командной строки при запуске файла
    parser = argparse.ArgumentParser()
    
    if len(sys.argv) == 1: 
        get_pop_recs() # Нет аргументов, выдаем треки по умолчанию
    
    else:
        parser.add_argument ('-user_id', '--user_id')
        parser.add_argument ('-top_k', '--top_k')
        namespace = parser.parse_args(sys.argv[2:])
        if namespace.user_id is None: 
            logger.info(f"Error, wrong parameters")
        elif namespace.top_k is None:
            get_user_recs(namespace.user_id)
        else: 
            get_user_recs(namespace.user_id, namespace.top_k)

        