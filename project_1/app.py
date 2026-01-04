import asyncio
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import aiohttp
import altair as alt
import pandas as pd
import requests
import streamlit as st


@st.cache_resource
def load_data(file):
    return pd.read_csv(file)


def get_current_weather_sync(city: str, api_key: str, units="metric"):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": units}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()  # упадёт при 4xx / 5xx

    return response.json()


async def get_current_weather_async(city: str, api_key: str, units="metric"):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": units}

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()


def run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def analyze_one_city(
    city_df: pd.DataFrame,
    window_days: int = 30,
) -> pd.DataFrame:
    city_df = city_df.sort_values("timestamp").copy()

    city_df["rolling_temperature"] = (
        city_df["temperature"].rolling(window=window_days, min_periods=1).mean()
    )

    seasonal_stats = (
        city_df.groupby(["city", "season"], as_index=False)["temperature"]
        .agg(season_mean="mean", season_std="std")
        .copy()
    )
    seasonal_stats["season_std"] = seasonal_stats["season_std"].fillna(0.0)

    city_df = city_df.merge(seasonal_stats, on=["city", "season"], how="left")
    city_df["lower_thr"] = city_df["season_mean"] - 2 * city_df["season_std"]
    city_df["upper_thr"] = city_df["season_mean"] + 2 * city_df["season_std"]
    city_df["is_anomaly"] = (city_df["temperature"] < city_df["lower_thr"]) | (
        city_df["temperature"] > city_df["upper_thr"]
    )
    return city_df


def run_analysis_sequential(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for _, city_df in df.groupby("city", sort=True):
        parts.append(analyze_one_city(city_df, window_days=window_days))

    return pd.concat(parts, ignore_index=True)


def run_analysis_parallel(
    df: pd.DataFrame, window_days: int, max_workers: int | None = None
) -> pd.DataFrame:
    cities = sorted(df["city"].unique())
    if not cities:
        return df.copy()

    cpu_count = os.cpu_count() or 1
    if max_workers is None:
        max_workers = min(len(cities), cpu_count)
    max_workers = max(1, int(max_workers))

    # "spawn" может перезапускать скрипт Streamlit в дочерних процессах
    # Для анализа используем "fork"
    try:
        ctx = mp.get_context("fork")
    except ValueError as e:
        raise RuntimeError(
            "Multiprocessing недоступен в текущем окружении (нет start method 'fork')."
        ) from e
    worker = partial(analyze_one_city, window_days=window_days)
    city_chunks = [df[df["city"] == city].copy() for city in cities]

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        results = list(executor.map(worker, city_chunks))

    return pd.concat(results, ignore_index=True)


month_to_season = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}

WINDOW_DAYS = 30

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Анализ температурных данных и мониторинг текущей температуры ")
st.markdown("---")
uploaded_file = st.sidebar.file_uploader("#### Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл c историческими данными о погоде")
else:
    df = load_data(uploaded_file)
    st.subheader("📊 Анализ исторических данных")

    cpu_count = os.cpu_count() or 1
    st.caption(
        f"Параметры:  \n"
        f"CPU (для multiprocessing): {cpu_count} | городов: {int(df['city'].nunique())} | строк: {len(df)} | окно: {WINDOW_DAYS} дней"
    )

    # Считаем сначала последовательно далее параллельно для сравнения скорости
    seq_time_start = time.perf_counter()
    df_stats = run_analysis_sequential(df, window_days=WINDOW_DAYS)
    seq_time = time.perf_counter() - seq_time_start
    seasonal_stats = df_stats[
        ["city", "season", "season_mean", "season_std"]
    ].drop_duplicates()
    anomalies_df = df_stats[df_stats["is_anomaly"]].copy()

    mp_time = None
    try:
        mp_time_start = time.perf_counter()
        run_analysis_parallel(df, window_days=WINDOW_DAYS, max_workers=cpu_count)
        mp_time = time.perf_counter() - mp_time_start
    except RuntimeError as e:
        st.warning(str(e))

    st.markdown("#### Сравнение скорости")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Seq", f"{seq_time:.3f}s")
    if mp_time is None:
        c2.metric(f"Multiprocessing ({cpu_count})", "n/a")
        c3.metric("Δ (seq - mp)", "n/a")
        c4.metric("Ускорение", "n/a")
    else:
        c2.metric(f"Multiprocessing ({cpu_count})", f"{mp_time:.3f}s")
        c3.metric("Δ (seq - mp)", f"{(seq_time - mp_time):.3f}s")
        c4.metric(
            "Ускорение",
            f"{(seq_time / mp_time) if mp_time else float('inf'):.2f}x",
        )

    city_for_plot = st.selectbox(
        "Город для графика:", sorted(df_stats["city"].unique())
    )
    city_view = df_stats[df_stats["city"] == city_for_plot].sort_values("timestamp")
    base = alt.Chart(city_view).encode(x=alt.X("timestamp:T", title="Дата"))
    line_temp = base.mark_line(color="#4C78A8").encode(
        y=alt.Y("temperature:Q", title="Температура (°C)"),
        tooltip=["timestamp:T", "temperature:Q"],
    )
    line_ma = base.mark_line(color="#72B7B2").encode(
        y="rolling_temperature:Q",
        tooltip=["timestamp:T", "rolling_temperature:Q"],
    )
    anomaly_points = (
        alt.Chart(city_view[city_view["is_anomaly"]])
        .mark_point(color="#E45756", size=35)
        .encode(
            x="timestamp:T",
            y="temperature:Q",
            tooltip=["timestamp:T", "temperature:Q", "season:N"],
        )
    )
    chart = (line_temp + line_ma + anomaly_points).properties(height=320).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Статистика по сезонам (mean и std) для каждого города")
    st.dataframe(
        seasonal_stats.sort_values(["city", "season"]).reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("#### Аномалии: температура вне mean ± 2σ (по сезону и городу)")
    st.write(f"Найдено аномалий: {len(anomalies_df)}")
    anomalies_by_city = (
        anomalies_df.groupby("city", as_index=False)["is_anomaly"]
        .count()
        .rename(columns={"is_anomaly": "anomalies"})
        .sort_values("anomalies", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(
        anomalies_by_city,
        use_container_width=True,
    )

    with st.expander("Показать строки-аномалии (по выбранному городу)"):
        st.dataframe(
            anomalies_df[anomalies_df["city"] == city_for_plot]
            .sort_values("timestamp")
            .reset_index(drop=True),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("📈 Мониторинг текущей температуры")

    with st.sidebar:
        selected_col = st.selectbox(
            "#### Выберите город для проверки текущей температуры:", df["city"].unique()
        )

        with st.form("api_key_form"):
            api_key = st.text_input(
                "#### Введите API ключ для получения текущей температуры:"
            )
            submitted = st.form_submit_button("Отправить")
    if not api_key:
        st.info("👈 Введите API ключ для получения текущей температуры")
    if api_key and submitted:
        try:
            sync_start = time.perf_counter()
            w_dict = get_current_weather_sync(selected_col, api_key)
            sync_time = time.perf_counter() - sync_start

            async_start = time.perf_counter()
            _ = run_async(get_current_weather_async(selected_col, api_key))
            async_time = time.perf_counter() - async_start

            st.markdown("#### Сравнение скорости запроса текущей температуры")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sync", f"{sync_time:.3f}s")
            c2.metric("Async", f"{async_time:.3f}s")
            c3.metric("Δ (sync - async)", f"{(sync_time - async_time):.3f}s")
            c4.metric(
                "Ускорение",
                f"{(sync_time / async_time) if async_time else float('inf'):.2f}x",
            )

            current_temp = w_dict["main"]["temp"]
            current_season = month_to_season[
                pd.to_datetime(w_dict["dt"], unit="s").month
            ]
            current_df = df_stats[
                (df_stats["city"] == selected_col)
                & (df_stats["season"] == current_season)
            ].iloc[0]
            lower_thr = current_df["lower_thr"]
            upper_thr = current_df["upper_thr"]
            is_anomaly = (current_temp < lower_thr) | (current_temp > upper_thr)
            if is_anomaly:
                st.error(
                    f"⚠️ Аномалия! Текущая температура {current_temp} °C выходит за пределы {lower_thr:.2f} °C - {upper_thr:.2f} °C для сезона {current_season} в городе {selected_col}."
                )
            else:
                st.success(
                    f"✅ Текущая температура {current_temp} °C в пределах нормы ({lower_thr:.2f} °C ; {upper_thr:.2f} °C) для сезона {current_season} в городе {selected_col}."
                )

        except (requests.HTTPError, RuntimeError) as e:
            st.error(f"Ошибка при получении данных о погоде: {e}")

# Напишу здесь краткие выводы по работе
# Расчет статистик был произведен двумя способами: последовательно и мультипроцессорно. Многопоточность не стал даже пробовать, тк это CPU-bound задача.
# Итогом стало то, что просто последовательное выполнение быстрее. Связано это с тем, что вычисления не слишком тяжелые и выигрыш от мультипроцессности теряется, тк тут затрачивается изначально больше ресурсов, чтоб разделить задачу на несколько процессов. (Конечно можно пробовать запускать с меньшим кол-вом ядер CPU, добавил в функцию отдельным параметром)
# Также реализован мониторинг текущей температуры с использованием как синхронного, так и асинхронного подходов. При первом запуске асинхронный подход показывает себя значительно быстрее (в 8 раз быстрее), но при повторных запусках (выбор другого города) разница нет. (В целом это понятно, тк у нас же один запрос к API в один момент времени). Выигрыш в самом начале думаю из за того, что основной поток не блокируется при асинхроне и streamlit может продолжать выполнять свои процессы.
# В итоге скажу, что стоит выбирать асинхронный режим, тк он быстрее при запуске приложения + будет всегда возможность поменять логику приложения и грузить сразу всю текущую погоду по всем городам. Там выигрыш асинхронного подхода будет очевиден.
