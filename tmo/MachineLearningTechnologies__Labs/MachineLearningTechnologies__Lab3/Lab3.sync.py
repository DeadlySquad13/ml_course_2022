# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# %% [markdown]
# # Лабораторная работа №3
# # Подготовка обучающей и тестовой выборки, кросс-валидация и подбор гиперпараметров на примере метода ближайших соседей
#
# Выполнил: **Пакало А. С., РТ5-61Б**

# %% [markdown]
# ## Задание
# Выберите набор данных (датасет) для решения задачи классификации или
# регрессии.
# - С использованием метода train_test_split разделите выборку на обучающую и
#   тестовую.
# - Обучите модель ближайших соседей для произвольно заданного гиперпараметра
# K.
#   Оцените качество модели с помощью подходящих для задачи метрик.
# - Произведите подбор гиперпараметра K с использованием GridSearchCV и/или
#   RandomizedSearchCV и кросс-валидации, оцените качество оптимальной модели.
#   Желательно использование нескольких стратегий кросс-валидации.
# - Сравните метрики качества исходной и оптимальной моделей.
 
# %% [markdown]
# ## Текстовое описание набора данных
# Для обучения по методу K ближайших соседей (KNN) был выбран датасет с
# классификацией типа звёзд c ресурса kaggle (Star Type Classification / NASA).
 
# В данном наборе данных присутствуют следующие столбцы:
# * Temperature — температура звезды в Кельвинах;
# * L (Luminosity) — светимость звезды в солнечных светимостях;
# * R (Radius) — радиус звезды в радиусах солнца;
# * A_M (Absolute Magnitude) — абсолютная звёздная величина;
# * Color — цвет света звезды;
# * Spectral_Class — спектральный класс звезды;
# * Type — тип звезды. Является целевым признаком и уже закодирован:
#   - Красный карлик — 0;
#   - Коричневый карлик — 1;
#   - Белый карлик — 2;
#   - Звезда из главной последовательности — 3;
#   - Супергигант — 4;
#   - Гипергигант — 5.

# Так как данных очень много, перед тем как приступить к анализу, проведем обзор данных и, возможно, потребуется их предобработка, чтобы датасет стал более удобным и пригодным к проведению исследования.
# 
# Таким образом исследование пройдет в 7 этапов:
# - загрузка данных,
# - проведение разведочного анализа данных и предобработка данных,
# - разделение на обучающую и тестовую выборку,
# - выбор метрики,
# - обучение модели,
# - подбор гиперпараметра,
# - сравнение значений метрики.

# %% [markdown]
# ## Импортирование необходимых библиотек, подготовка окружения

# %%
# Основные библиотеки.
import numpy as np
import pandas as pd

# Визуализция.
import matplotlib.pyplot as plt
import seaborn as sns

# Для матрицы взаимодействий.
from scipy import sparse
# Для разбития выборки.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from DataFrameOneHotEncoder import DataFrameOneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Отрисовка статуса выполнения.
from tqdm.notebook import tqdm

# Типизация.
from typing import List

# Вывод данных.
from IPython.display import display, Markdown
def printmd(message):
    display(Markdown(message))


# Конфигурация визуализации.
%matplotlib inline
sns.set(style='ticks')

# %% [markdown]
# ## Загрузка данных
# ### Считываем данные из .csv
# Загрузим файлы датасета в помощью библиотеки Pandas.
# 
# Не смотря на то, что файлы имеют расширение txt они представляют собой данные
# в формате [CSV](https://ru.wikipedia.org/wiki/CSV). Часто в файлах такого
# формата в качестве разделителей используются символы ",", ";" или табуляция.
# Поэтому вызывая метод read_csv всегда стоит явно указывать разделитель данных
# с помощью параметра sep. Чтобы узнать какой разделитель используется в файле
# его рекомендуется предварительно посмотреть в любом текстовом редакторе.

# %%
stars = pd.read_csv('data/Stars.csv')

# %% [markdown]
# ## Проведение разведочного анализа данных. Построение графиков, необходимых для понимания структуры данных. Анализ и предобработка данных.

# %% [markdown]
# Размеры датасета: (строки, колонки).

# %%
stars.shape

# %% [markdown]
# Общий вид данных таблицы:

# %%
stars.head()

# %% [markdown]
# Список колонок:

# %%
stars.columns

# %% [markdown]
# Список колонок с типами данных:

# %%
stars.dtypes

# %% [markdown]
# Как видно, все данные приведены к адекватному типу данных.


# %% [markdown]
# ## Предобработка данных

# %%
# Извлекаем целевой признак.
TARGET_COL_NAMES = ['Type']
star_types = stars[TARGET_COL_NAMES]

star_features = stars.drop(columns=TARGET_COL_NAMES)
display(star_types, star_features)

# %%
# Перед использованием модели закодируем категориальные признаки с помощью
# one-hot encoding, где каждое уникальное значение признака становится новым
# признаком. Это позволяет избежать фиктивного отношения порядка.

# %%
categorical_pipeline = Pipeline([
    ( 'one-hot', DataFrameOneHotEncoder(handle_unknown='ignore') )
])
CATEGORICAL_COL_NAMES = ['Color', 'Spectral_Class']
# Returns tuple: (2d-array with columns?, shape).
caterogical_star_features = categorical_pipeline.fit_transform(stars[CATEGORICAL_COL_NAMES]),
caterogical_star_features = pd.DataFrame(caterogical_star_features[0])

# %% [markdown]
# Нам также потребуется масштабировать данные для адекватной работы модели.

# %%
numerical_pipeline = Pipeline([
    ( 'scaler', StandardScaler() )
])


numerical_star_features = star_features.drop(columns=CATEGORICAL_COL_NAMES)
numerical_star_features_transformed = numerical_pipeline.fit_transform(numerical_star_features)
# Массив переводим обратно в датафрейм.
numerical_star_features_transformed = pd.DataFrame(numerical_star_features,
                                                   columns=numerical_star_features.columns)
numerical_star_features_transformed

# %%
NUMERICAL_COL_NAMES = list(filter(lambda feature:
        feature not in
        CATEGORICAL_COL_NAMES,
    star_features.columns))

preprocessor = ColumnTransformer([
    ( 'numerical', numerical_pipeline, NUMERICAL_COL_NAMES ),
    ( 'categorical', categorical_pipeline, CATEGORICAL_COL_NAMES)
])

star_features_preprocessed = preprocessor.fit_transform(star_features)

# %% [markdown]
# ## С использованием метода train_test_split разделите выборку на обучающую и тестовую.

# %%

star_features_train: pd.DataFrame
star_features_test: pd.DataFrame
star_types_train: pd.Series
star_types_test: pd.Series

# Параметр random_state позволяет задавать базовое значение для генератора
# случайных чисел. Это делает разбиение неслучайным. Если задается параметр
# random_state то результаты разбиения будут одинаковыми при различных
# запусках. На практике этот параметр удобно использовать для создания
# "устойчивых" учебных примеров, которые выдают одинаковый результат при
# различных запусках.
RANDOM_STATE_SEED = 1

star_features_train, star_features_test, star_types_train, star_types_test = train_test_split(
    star_features, star_types, random_state=RANDOM_STATE_SEED)

# %% [markdown]
# Общий вид обучающей выборки:

# %%
pd.DataFrame(star_features_preprocessed)

# %%
display(star_features_train.head(), star_types_train.head())

# %% [markdown]
# Общий вид тестовой выборки:

# %%
display(star_features_test.head(), star_types_test.head())

# %% [markdown]
# ## Обучение и оценка модели ближайших соседей для произвольно заданного гиперпараметра
# В классической модели ближайших соседей гиперпараметром является количество
# соседей. Зададим его в константе N_NEIGHBORS.

# %%
from sklearn.metrics import mean_absolute_error

N_NEIGHBORS = 5
# В KNN Наиболее часто используется Евклидова метрика, поэтому для определения веса
#   соседей выберем параметр 'distance'.
knn_classifier = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance')

knn_pipeline = Pipeline([
    ( 'preprocess', preprocessor ),
    ( 'model', knn_classifier ),
])

star_types_predicted = knn_pipeline.fit(star_features_train,
                   star_types_train.values.ravel()).predict(star_features_test)
mean_absolute_error(star_types_test, star_types_predicted)

# %% [markdown]
# ## Произведите подбор гиперпараметра K с использованием GridSearchCV и/или RandomizedSearchCV и кросс-валидации, оцените качество оптимальной модели.
# Желательно использование нескольких стратегий кросс-валидации.

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import SCORERS

params = { 'model__n_neighbors': np.arange(1, 30, 1) }

display(sorted(SCORERS.keys()))
grid_search = GridSearchCV(knn_pipeline, params, scoring='neg_mean_absolute_error')

grid_search.fit(star_features_train, star_types_train.values.ravel())
display(grid_search.best_score_, grid_search.best_params_)


# %% [markdown]
# ## Сравните метрики качества исходной и оптимальной моделей.
