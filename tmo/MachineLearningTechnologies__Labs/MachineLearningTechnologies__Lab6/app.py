import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

@st.cache(suppress_st_warning=True)
def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('data/Walmart.csv', sep=",", nrows=500)
    return data

preprocessor = Pipeline([
    ( 'scaler', MinMaxScaler(feature_range=(0, 1)) )
])

@st.cache(suppress_st_warning=True)
def preprocess_data(data):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''

    walmart = pd.DataFrame.copy(data)
    # %% [markdown]
    # ## Предобработка данных
    # С признаком, хранящий дату, будет легче работать, если разбить его на три
    # отдельные признака: неделя, месяц и год. Можем позволить оставить наименьшей
    # единицей неделю, так как нам даны лишь *недельные* продажи магазина.

    # %%
    # Разделение поля Date
    walmart['Date'] = pd.to_datetime(walmart['Date'], dayfirst=True)
    walmart['Week'] = walmart['Date'].dt.isocalendar().week.astype('int64')
    walmart['Month'] = walmart['Date'].dt.month
    walmart['Year'] = walmart['Date'].dt.year

    # Удаление столбца Date
    walmart = walmart.drop(columns=['Date'])

    # Перестановка столбцов
    walmart = walmart[['Store', 'Week', 'Month', 'Year', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales']]

    # %% [markdown]
    # На основе корреляционной матрицы можем сделать следующие выводы:
    # 
    # - Все нецелевые признаки имеют слабую связь с целевым (Weekly_Sales). Сильнее
    # всего коррилиует признак Store;
    # - Признаки Week и Month практически линейно зависимы друг от друга. Оставим
    # только признак Week;
    # - Признаки Year и Fuel_Price тоже сильно коррелируют друг с другом. Оставим
    # признак Fuel_Price.

    # %%
    # Удаление лишних колонок.
    walmart_without_linear_correlates = walmart.drop(columns=['Month', 'Year'])


    # %% [markdown]
    # ## Очистка выбросов.
    # Очистка столбца Temperature.
    Q1, Q3 = walmart['Temperature'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    min_limit = Q1 - 1.5*IQR
    max_limit = Q3 + 1.5*IQR

    walmart_without_outliers = walmart_without_linear_correlates[
        (walmart_without_linear_correlates['Temperature']
        > min_limit) & (walmart_without_linear_correlates['Temperature'] < max_limit)
    ]
    # Очистка столбца Unemployment.
    Q1, Q3 = walmart['Unemployment'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    min_limit = Q1 - 1.3*IQR
    max_limit = Q3 + 1.3*IQR

    walmart_without_outliers = walmart_without_outliers[(walmart_without_outliers['Unemployment'] > min_limit)
                                                  & (walmart_without_outliers['Unemployment'] < max_limit)]
    # Очистка столбца Weekly_Sales.
    Q1, Q3 = walmart['Weekly_Sales'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    min_limit = Q1 - 1.5*IQR
    max_limit = Q3 + 1.5*IQR

    walmart_without_outliers = walmart_without_outliers[(walmart_without_outliers['Weekly_Sales'] > min_limit)
                                                  & (walmart_without_outliers['Weekly_Sales'] < max_limit)]
    # %% [markdown]
    # ## Масштабирование
    # Нам также потребуется масштабировать данные для адекватной работы моделей (и
    # линейные , и SVM работают лучше, если  признаки представлены в одном
    # масштабе).
    walmart_transformed = preprocessor.fit_transform(walmart_without_outliers)
    # Массив переводим обратно в датафрейм.
    walmart_transformed = pd.DataFrame(walmart_transformed,
                                                columns=walmart_without_outliers.columns)
    walmart_transformed

    # %%
    # Извлекаем целевой признак.
    TARGET_COL_NAMES = ['Weekly_Sales']
    walmart_weekly_sales = walmart_transformed[TARGET_COL_NAMES]

    walmart_features = walmart_transformed.drop(columns=TARGET_COL_NAMES)

    # %% [markdown]
    # ## С использованием метода train_test_split разделите выборку на обучающую и тестовую.

    walmart_features_train: pd.DataFrame
    walmart_features_test: pd.DataFrame
    walmart_weekly_sales_train: pd.Series
    walmart_weekly_sales_test: pd.Series

    # Параметр random_state позволяет задавать базовое значение для генератора
    # случайных чисел. Это делает разбиение неслучайным. Если задается параметр
    # random_state то результаты разбиения будут одинаковыми при различных
    # запусках. На практике этот параметр удобно использовать для создания
    # "устойчивых" учебных примеров, которые выдают одинаковый результат при
    # различных запусках.
    RANDOM_STATE_SEED = 1

    walmart_features_train, walmart_features_test, walmart_weekly_sales_train, walmart_weekly_sales_test = train_test_split(
        walmart_features, walmart_weekly_sales, random_state=RANDOM_STATE_SEED)

    return (walmart_features_train, walmart_features_test,
            walmart_weekly_sales_train, walmart_weekly_sales_test)


st.header('Обучение модели ближайших соседей')

data_load_state = st.text('Загрузка данных...')
data = load_data()
data_load_state.text('Данные загружены!')

#Количество записей
data_len = data.shape[0]

if st.checkbox('Показать корреляционную матрицу'):
    fig1, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)

n_splits_slider = st.slider('n_splits', min_value=1, max_value=10, value=5, step=1)

st.write('Количество строк в наборе данных - {}'.format(data_len))

walmart_features_train, walmart_features_test, walmart_weekly_sales_train, walmart_weekly_sales_test = preprocess_data(data)

linear_regression = LinearRegression(n_jobs=3)

linear_regression_pipeline = Pipeline([
    ( 'preprocess', preprocessor ),
    ( 'model', linear_regression ),
])

walmart_weekly_sales_predicted = linear_regression_pipeline.fit(walmart_features_train,
                   walmart_weekly_sales_train).predict(walmart_features_test)

mae = mean_absolute_error(walmart_weekly_sales_predicted, walmart_weekly_sales_test)
mse = mean_squared_error(walmart_weekly_sales_predicted, walmart_weekly_sales_test)
r2 = r2_score(walmart_weekly_sales_predicted, walmart_weekly_sales_test)


polynomial_features = PolynomialFeatures()

polynomial_regression_estimator = Pipeline([
    ( 'preprocess', preprocessor ),
    ( 'poly_features', polynomial_features ),
    ( 'model', linear_regression ),
])

parameters_to_tune = { 'poly_features__degree': np.arange(1, 6, 1) }

scoring = (
    'neg_mean_absolute_error',
    'neg_mean_squared_error',
    'r2',
)

polynomial_regression_pipeline = RandomizedSearchCV(polynomial_regression_estimator,
                                                    parameters_to_tune,
                                                    n_iter=5,
                                                    scoring=scoring,
                                                    cv=KFold(n_splits=n_splits_slider),
                                                    refit=scoring[2]
                                                   )

polynomial_regression_pipeline.fit(walmart_features_train,
                   walmart_weekly_sales_train)

# add_metrics_data_from_search_results(polynomial_regression_pipeline.cv_results_,
                                     # model='PolynomialRegression')


st.subheader('Оценка качества модели')
st.write('Значения mae, mse, r2 метрик')
linear_metrics = pd.DataFrame([mae, mse, r2], ['mae', 'mse', 'r2'])
st.bar_chart(linear_metrics)
polynomial_metrics = pd.DataFrame([polynomial_regression_pipeline.best_score_,
        polynomial_regression_pipeline.best_params_['poly_features__degree']],
                                  ['best_score', 'degree'])
st.bar_chart(polynomial_metrics)
