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
from IPython.core.display import display_markdown
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
from IPython.display import display


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
from typing import Union
def display_df_shape(df: Union[pd.DataFrame, pd.Series]):
    display_markdown(
        f'- Кол-во рядов: {df.shape[0]}\n- Кол-во колонок: {df.shape[1]}',
        raw=True)


def display_split_result(split: Union[pd.DataFrame, pd.Series]):
    """Display result of the train_test_split.

    :param split: one of the splits to display.
    :type split: pd.DataFrame, pd.Series
    """

    display(split.head(1))
    display_df_shape(split)
    
    
display_split_result(star_features_train)
display_split_result(star_types_train)

# %% [markdown]
# Общий вид тестовой выборки:

# %%
display_split_result(star_features_test)
display_split_result(star_types_test)

# %% [markdown]
# ## Обучение и оценка модели ближайших соседей для произвольно заданного гиперпараметра
# В классической модели ближайших соседей гиперпараметром является количество
# соседей. Зададим его в константе N_NEIGHBORS.

# %%
from sklearn.metrics import accuracy_score

N_NEIGHBORS = 12
# В KNN Наиболее часто используется Евклидова метрика, поэтому для определения веса
#   соседей выберем параметр 'distance'.
knn_classifier = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance')

knn_pipeline = Pipeline([
    ( 'preprocess', preprocessor ),
    ( 'model', knn_classifier ),
])

star_types_predicted = knn_pipeline.fit(star_features_train,
                   star_types_train.values.ravel()).predict(star_features_test)
# knn_pipeline.score(star_types_predicted, star_types_test.values.ravel())
accuracy_score(star_types_predicted, star_types_test)

# mean_absolute_error(star_types_test, star_types_predicted)
# neg_mean_absolute_error(star_types_test, star_types_predicted)

# %% [markdown]
# ## Произведите подбор гиперпараметра K с использованием GridSearchCV и/или RandomizedSearchCV и кросс-валидации, оцените качество оптимальной модели.
# Желательно использование нескольких стратегий кросс-валидации.

# %% [markdown]
### С помощью GridSearchCV

# %%
%%time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import SCORERS

params = { 'model__n_neighbors': np.arange(1, 30, 1) }

display(sorted(SCORERS.keys()))
grid_search = GridSearchCV(knn_pipeline, params, cv=KFold(n_splits=10),
                           scoring='accuracy')

grid_search.fit(star_features_train, star_types_train.values.ravel())
display(grid_search.best_score_, grid_search.best_params_)

# %% [markdown]
### С помощью RandomizedSearchCV

# %%
%%time
from sklearn.model_selection import RandomizedSearchCV

randomized_search = RandomizedSearchCV(knn_pipeline, params,
                                       cv=KFold(n_splits=10),
                                       scoring='accuracy')

randomized_search.fit(star_features_train, star_types_train.values.ravel())
display(grid_search.best_score_, grid_search.best_params_)

# %% [markdown]
# ## Сравните метрики качества исходной и оптимальной моделей

# %%
def plot_hyperparam_curve(models):
    plt.title('Сравнение метрик моделей')
    plt.xlabel('Метрика')
    plt.ylabel('Значение')
    delta = np.ones(3) * -0.2
    bar_width = 0.2

    for model_name, (metrics, metric_values) in models.items():
        plt.bar(metrics + delta, metric_values, label=model_name,
                width=bar_width)
        delta += bar_width


    ax = plt.figure().add_subplot()
    ax.set_xticklabels( ('2011-Jan-4', '2011-Jan-5', '2011-Jan-6') )
    plt.legend(loc="best")
    return plt


from typing import Dict

def grouped_bar_chart(ax, data: Dict[str, List[Union[int, np.nan]]],
                      tick_labels, colors=None, total_width=0.8,
             single_width=1):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax: matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: Dict[str, List[Union[int, np.nan]]]
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[ 1, 2,      3      ],
            "y":[ 1, np.nan, 3      ],
            "z":[ 1, 2,      np.nan ],
        }

    tick_labels: list
        Labels which will be displayed under each group.

        Example:
        tick_labels = ['Accuracy', 'F1', 'Precision']

        Note:
        Length should be the same as number of groups.
        

    colors: array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width: float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    # Number of bars per group.
    n_bars = len(data)

    # The width of a single bar.
    bar_width = total_width / n_bars

    tick_locations = np.arange(len(tick_labels))

    # Iterate over all data.
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar.
        tick_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # One bar plot consists of multiple rectangles.
        rects = ax.bar(tick_locations + tick_offset, values, width=bar_width * single_width,
                     color=colors[i % len(colors)],
                     label=name)
        # Add a handle to the last drawn bar, which we'll need for the legend.
        # bars.append(bars[0])

        ax.bar_label(rects)

    ax.set_xticks(tick_locations, labels=tick_labels)


def show_metrics_grouped_bar_chart(metrics_data: Dict[str, Dict[str, int]]):
    """ Creates grouped bar chart for metrics.
    :param metrics_data: a dictionary of metrics and their values for each
        model.
    :type metrics_data: Dict[str, Dict[str, int]]

        Example:

        metrics_data = {
            'GridSearchCV': { 
                'Accuracy': 1,
                'a': 2,
                'b': 3,
            },
            'RandomSearchCV': {
                'Roc': 2,
                'a': 3,
                'b': 4,
            },
        }

    """
    plt.figure(figsize=(4,1))
    # width = 15
    # height = 15
    # plt.rcParams['figure.figsize'] = width, height

    fig, ax = plt.subplots()

    # Extracting unique metric names from our data.
    tick_labels = set()
    for model_metrics in metrics_data.values():
        tick_labels.update(list(model_metrics.keys()))

    """ Converting our dictionary to a `fully filled` dictionary of lists:
    data = {
        'GridSearchCV': [ 
            1, # was 'Accuracy': 1,
            2, # was 'a': 2,
            3, # was 'b': 3,
            None, # didn't have value for 'Roc';
        ],
        'RandomSearchCV': [
            None, # didn't have value for 'Accuracy',
            3, # was 'a': 3,
            4, # was 'b': 4,
            2, # was 'Roc': 2;
        ],
    }
    """
    filled_metrics_data = {}

    for model in metrics_data.keys():
        filled_metrics_data[model] = []
        for metric in tick_labels:
            filled_metrics_data[model].append(metrics_data[model].get(metric)
                                              or np.nan)
            
    
    grouped_bar_chart(ax, filled_metrics_data, tick_labels=tick_labels,
                      total_width=.8, single_width=.9)

    plt.title('Сравнение метрик качества моделей')
    plt.xlabel('Метрика')
    plt.ylabel('Значение метрики')

    ax.legend(title='Модели')
    plt.show()


metrics_data = {
    'GridSearchCV': { 
        'Accuracy': 1,
        'a': 2,
        'b': 3,
    },
    'RandomSearchCV': {
        'Roc': 2,
        'a': 3,
        'b': 4,
    },
}

metrics_data['GridSearchCV']['Accuracy'] = 1

show_metrics_grouped_bar_chart(metrics_data)
