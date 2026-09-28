"""Построение признаков (фич) из событий внутри 24-часового окна наблюдения.

Каждая функция-фичер принимает DataFrame событий (уже отфильтрованных по окну)
и meta-таблицу (train или test), возвращает DataFrame с фичами по cookie_id.

Группы фич:
- basic_feats: размер сессии и базовое разнообразие
- timing_feats: тайминг событий, периодичность, активные часы, max-burst
- event_mix_feats: распределение по типам событий
- platform_feats: распределение по платформам (с нормализацией регистра)
- search_feats: поведение в поиске
- diversity_feats: разнообразие контента и энтропия категорий
- pointer_feats: фичи по координатам курсора
- ua_feats: фичи по User-Agent (длина, ботовые паттерны, семейство браузера)
- meta_feats: возраст куки и временные метки окна
- behavior_feats: капча, логин, повторы item, транзакции
- bot_flags: ручные эвристические флаги
- cross_cookie_feats: частоты UA/item и сглаженный bot-rate по train

- cross_cookie_feats: частоты UA/item в train+test (БЕЗ лейблов — это частота, не утечка)

Все функции возвращают DataFrame с колонкой cookie_id и дополнительными фичами.
"""
import re

import numpy as np
import pandas as pd

# Полный список типов событий (в данных встречаются только эти 10).
# Фиксируем явно, чтобы порядок колонок в one-hot был стабильный
# и пропущенные типы (когда у куки нет такого события) давали 0.
EVENT_NAMES = [
    "search_results_view", "item_view", "photo_swipe", "favorite_add",
    "seller_page_view", "contact_phone_show", "contact_chat_open",
    "contact_message_sent", "captcha_shown", "login",
]

# Платформы встречаются в данных в разных регистрах (web/Web/WEB, android/Android/ANDROID),
# поэтому везде приводим к нижнему регистру и группируем в 4 канонических.
PLATFORMS = ["web", "android", "ios", "desktop"]

# Паттерны "ботовых" User-Agent. Это эвристика: если UA содержит headless/selenium/curl и т.п.,
# то это явный признак автоматизации. Используется в ua_feats и bot_flags.
UA_BOT_RE = re.compile(
    r"(?:headless|phantom|selenium|playwright|puppeteer|"
    r"curl|wget|python-requests|httpclient|scrapy|node-fetch|"
    r"http\.client|okhttp|apache-httpclient|java/)",
    re.IGNORECASE,
)


def events_in_window(ev_all, meta):
    """Фильтрует события строго по окну наблюдения cookie.

    По условию задачи признаки должны быть доступны на момент окончания окна,
    поэтому используем полуинтервал [window_start, window_end).

    Аргументы:
        ev_all: все события (из events.csv.gz)
        meta: meta-таблица (train или test) с колонками cookie_id, window_start_ts, window_end_ts

    Возвращает: DataFrame событий, в котором каждой строке приклеены границы окна.
    """
    m = ev_all.merge(
        meta[["cookie_id", "window_start_ts", "window_end_ts"]],
        on="cookie_id", how="inner",
    )
    return m[(m.event_ts >= m.window_start_ts) & (m.event_ts < m.window_end_ts)]


def basic_feats(ev, meta):
    """Базовые счётчики: размер сессии и разнообразие сущностей.

    Колонки:
        n_events: сколько событий у куки в окне
        n_unique_item/cat/loc/seller: сколько разных объявлений/категорий/локаций/продавцов

    fillna(0) нужен для кук, у которых вообще нет событий в окне.
    """
    g = ev.groupby("cookie_id")
    f = pd.DataFrame({
        "n_events": g.size(),
        "n_unique_item": g.item_id.nunique(),
        "n_unique_cat": g.item_category.nunique(),
        "n_unique_loc": g.item_location.nunique(),
        "n_unique_seller": g.seller_type.nunique(),
    })
    return meta[["cookie_id"]].merge(f.reset_index(), on="cookie_id", how="left").fillna(0)


def timing_feats(ev, meta):
    """Тайминговые фичи — самая большая группа.

    Подгруппы:
    1) dt-статистики по интервалам между соседними событиями куки (mean/std/median/квантили).
       Боты обычно кликают быстрее и равномернее людей, поэтому std у них ниже.
    2) dt_cv: коэффициент вариации интервалов (std/mean). Боты часто имеют низкий cv.
    3) dt_periodic5 / dt_periodic15: доля интервалов, попавших в узкий коридор вокруг
       медианы (медиана +- 5% и +- 15%). Боты с периодическими запросами дают высокую долю.
    4) log_mean / log_std: log1p от среднего и std — чтобы деревья лучше работали
       с тяжёлыми хвостами распределений.
    5) session_span: разница между самым ранним и самым поздним событием в окне (в секундах).
       session_span_share = session_span / 86400 (доля суток, в течение которой шла активность).
    6) Распределение по часам (24 бинарные колонки hour_0..hour_23) и энтропия этого
       распределения (равномерное = высокая энтропия, точечное = низкая).
    7) Доля "ночных" событий (20-05) и "рабочих" (09-17).
    8) max_burst_60s: максимальное число событий за 60 секунд внутри сессии.
       Боты часто делают короткие всплески активности.
    """
    # сортируем по cookie+времени, чтобы diff дал соседние события одной куки
    ev = ev.sort_values(["cookie_id", "event_ts"]).reset_index(drop=True)
    ev["dt"] = ev.groupby("cookie_id")["event_ts"].diff().dt.total_seconds()

    # Доля dt, попавших в коридор +-tol от своей медианы. Признак "почти равных интервалов".
    def periodic(s, tol):
        s = s.dropna()
        if len(s) < 3:
            return 0.0
        med = float(s.median())
        if med <= 0 or med > 600:
            return 0.0  # медиана > 10 минут - слишком редко для "периодичности"
        return float(((s >= med * (1 - tol)) & (s <= med * (1 + tol))).mean())

    g = ev.groupby("cookie_id")["dt"]
    dt_f = pd.DataFrame({
        "n_dt":      g.apply(lambda s: int(s.notna().sum())),
        "dt_mean":   g.mean(),
        "dt_std":    g.std(),
        "dt_min":    g.min(),
        "dt_q25":    g.quantile(0.25),
        "dt_median": g.median(),
        "dt_q75":    g.quantile(0.75),
        "dt_q95":    g.quantile(0.95),
        "dt_max":    g.max(),
    }).reset_index()
    dt_f["dt_cv"]         = g.apply(lambda s: float(s.std()/s.mean()) if s.mean() and s.mean() > 0 else 0.0).values
    dt_f["dt_periodic5"]  = g.apply(lambda s: periodic(s, 0.05)).values
    dt_f["dt_periodic15"] = g.apply(lambda s: periodic(s, 0.15)).values
    dt_f["dt_log_mean"]   = np.log1p(dt_f["dt_mean"].fillna(0))
    dt_f["dt_log_std"]    = np.log1p(dt_f["dt_std"].fillna(0))

    # длительность сессии в секундах
    span = ev.groupby("cookie_id")["event_ts"].agg(
        session_span=lambda s: (s.max() - s.min()).total_seconds(),
    ).reset_index()

    # распределение по часам
    ev["hour"] = ev.event_ts.dt.hour
    hour_counts = (
        ev.groupby(["cookie_id", "hour"]).size()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    p = hour_counts.astype(float).div(
        hour_counts.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0)
    # энтропия Шеннона через numpy (избегаем pandas-багов с унарным минусом на частичных Series)
    arr = p.to_numpy(dtype=float)
    hour_entropy = pd.DataFrame({
        "cookie_id": p.index,
        "hour_entropy": -(arr * np.log(np.clip(arr, 1e-9, None))).sum(axis=1),
    })
    share_night = pd.DataFrame({
        "cookie_id": p.index,
        "share_night": p[[20, 21, 22, 23, 0, 1, 2, 3, 4, 5]].sum(axis=1).values,
    })
    share_business = pd.DataFrame({
        "cookie_id": p.index,
        "share_business": p[[9, 10, 11, 12, 13, 14, 15, 16, 17]].sum(axis=1).values,
    })
    hour_counts.columns = ["hour_" + str(h) for h in hour_counts.columns]
    hour_counts = hour_counts.reset_index()

    # max burst за 60 секунд: для каждой куки - сколько максимум событий помещается
    # в любое 60-секундное окно. Считаем sliding window через два указателя.
    ts_int = ev.event_ts.values.astype("datetime64[s]").astype(np.int64)
    groups = ev.groupby("cookie_id").indices

    def max_burst(idx):
        if len(idx) < 2:
            return float(len(idx))
        ts = np.sort(ts_int[idx])
        best, j = 1, 0
        for i in range(len(ts)):
            while ts[i] - ts[j] > 60:
                j += 1
            best = max(best, i - j + 1)
        return float(best)

    burst = pd.DataFrame({
        "cookie_id": list(groups.keys()),
        "max_burst_60s": [max_burst(idx) for idx in groups.values()],
    })

    f = meta[["cookie_id"]].merge(dt_f, on="cookie_id", how="left")
    f = f.merge(span, on="cookie_id", how="left")
    f = f.merge(hour_counts, on="cookie_id", how="left")
    f = f.merge(hour_entropy, on="cookie_id", how="left")
    f = f.merge(share_night, on="cookie_id", how="left")
    f = f.merge(share_business, on="cookie_id", how="left")
    f = f.merge(burst, on="cookie_id", how="left")
    f["session_span"] = f["session_span"].fillna(0)
    f["session_span_share"] = f["session_span"] / 86400.0
    return f


def event_mix_feats(ev, meta):
    """Распределение событий по типам + производные признаки.

    Колонки cnt_<event>: сколько раз кука сделала событие данного типа.
    Колонки share_<event>: доля данного события в общем числе событий куки.
    n_event_types: сколько разных типов событий вообще встречалось.
    n_engagement: сумма "вовлекающих" событий (contact + favorite_add).
    n_view: сумма "просмотровых" (item_view + search_results_view).
    engagement_per_view: отношение вовлечения к просмотрам. У людей обычно > 0,
        у ботов часто ближе к 0 (они смотрят, но не вовлекаются).
    """
    enc = ev.groupby(["cookie_id", "event_name"]).size().unstack(fill_value=0)
    enc = enc.reindex(columns=EVENT_NAMES, fill_value=0)
    enc.columns = ["cnt_" + c for c in enc.columns]
    total = enc.sum(axis=1).replace(0, np.nan)
    shares = enc.div(total, axis=0).fillna(0)
    shares.columns = ["share_" + c for c in shares.columns]
    n_types = (enc > 0).sum(axis=1).rename("n_event_types")

    f = meta[["cookie_id"]].merge(enc.reset_index(), on="cookie_id", how="left").fillna(0)
    f = f.merge(shares.reset_index(), on="cookie_id", how="left").fillna(0)
    f = f.merge(n_types.reset_index(), on="cookie_id", how="left").fillna(0)

    eng = (f["cnt_contact_phone_show"] + f["cnt_contact_chat_open"]
           + f["cnt_contact_message_sent"] + f["cnt_favorite_add"]).rename("n_engagement")
    view = (f["cnt_item_view"] + f["cnt_search_results_view"]).rename("n_view")
    f["engagement_per_view"] = (eng / view.replace(0, np.nan)).astype(float)
    return f


def platform_feats(ev, meta):
    """Распределение по платформам (с приведением регистра к нижнему).

    plat_<p>: число событий с платформой p.
    plat_share_<p>: доля.
    n_platforms: сколько разных платформ использовала кука (1 = консистентный клиент).

    В данных платформы идут в разных регистрах (WEB/Web/web, ANDROID/Android/android и т.д.),
    поэтому приводим к нижнему регистру и маппим в 4 канонических значения.
    """
    plat = ev["platform"].astype(str).str.lower()
    enc = (
        pd.DataFrame({"cookie_id": ev.cookie_id, "platform": plat})
        .groupby(["cookie_id", "platform"]).size()
        .unstack(fill_value=0)
        .reindex(columns=PLATFORMS, fill_value=0)
    )
    enc.columns = ["plat_" + c for c in enc.columns]
    total = enc.sum(axis=1).replace(0, np.nan)
    shares = enc.div(total, axis=0).fillna(0)
    shares.columns = ["plat_share_" + c for c in shares.columns]
    f = meta[["cookie_id"]].merge(enc.reset_index(), on="cookie_id", how="left").fillna(0)
    f = f.merge(shares.reset_index(), on="cookie_id", how="left").fillna(0)
    nonzero = (enc > 0).sum(axis=1).rename("n_platforms")
    f = f.merge(nonzero.reset_index(), on="cookie_id", how="left").fillna(1)
    return f


def search_feats(ev, meta):
    """Фичи по поведению в поиске.

    n_search: сколько событий содержат search_query.
    n_search_page: сколько событий содержат search_page.
    share_search_with_query: доля страниц поиска, где был и запрос.
    n_unique_queries: сколько разных запросов в окне.
    query_len_mean/max/std: статистики длины запроса (люди пишут длиннее).
    page_mean/max/std: глубина просмотра (боты часто смотрят много страниц подряд).
    """
    df = ev.copy()
    df["has_query"] = df.search_query.notna().astype(int)
    df["has_page"] = df.search_page.notna().astype(int)
    g = df.groupby("cookie_id")
    f = pd.DataFrame({"n_search": g.has_query.sum(), "n_search_page": g.has_page.sum()})
    f["share_search_with_query"] = f["n_search"] / f["n_search_page"].replace(0, np.nan)
    f["n_unique_queries"] = g.search_query.nunique()
    qlen = df.assign(q=df.search_query.fillna("").str.len()).groupby("cookie_id")["q"].agg(
        query_len_mean="mean", query_len_max="max", query_len_std="std",
    )
    f = f.merge(qlen, on="cookie_id", how="left")
    pg = df.dropna(subset=["search_page"]).groupby("cookie_id")["search_page"].agg(
        page_mean="mean", page_max="max", page_std="std",
    )
    f = f.merge(pg, on="cookie_id", how="left")
    return meta[["cookie_id"]].merge(f.reset_index(), on="cookie_id", how="left").fillna(0)


def diversity_feats(ev, meta):
    """Разнообразие контента и энтропия категорий.

    uniq_<field>: сколько уникальных значений field у этой куки.
    uniq_<field>_per_event: то же, делённое на число событий (нормировка).
    cat_entropy: энтропия распределения категорий просмотренных объявлений.
        У людей обычно выше (смотрят разное), у ботов ниже (метут одно).
    """
    g = ev.groupby("cookie_id")
    f = pd.DataFrame({
        "uniq_item":   g.item_id.nunique(),
        "uniq_cat":    g.item_category.nunique(),
        "uniq_loc":    g.item_location.nunique(),
        "uniq_seller": g.seller_type.nunique(),
        "uniq_query":  g.search_query.nunique(),
    }).reset_index()
    n_ev = g.size().rename("n_events")
    f = f.merge(n_ev.reset_index(), on="cookie_id", how="left")
    for c in ["uniq_item", "uniq_cat", "uniq_loc", "uniq_seller", "uniq_query"]:
        f[c + "_per_event"] = f[c] / f["n_events"].replace(0, np.nan)

    # энтропия по категориям (Шеннон)
    cat_ev = ev.dropna(subset=["item_category"])
    if len(cat_ev):
        cat_counts = cat_ev.groupby(["cookie_id", "item_category"]).size().unstack(fill_value=0)
        p = cat_counts.astype(float).div(
            cat_counts.sum(axis=1).replace(0, np.nan), axis=0
        ).fillna(0)
        arr = p.to_numpy(dtype=float)
        ent = pd.DataFrame({
            "cookie_id": cat_counts.index,
            "cat_entropy": -(arr * np.log(np.clip(arr, 1e-9, None))).sum(axis=1),
        })
        f = f.merge(ent, on="cookie_id", how="left")
    f["cat_entropy"] = f["cat_entropy"].fillna(0) if "cat_entropy" in f.columns else 0.0
    return meta[["cookie_id"]].merge(f, on="cookie_id", how="left").fillna(0)


def pointer_feats(ev, meta):
    """Фичи по координатам курсора.

    share_with_ptr: доля событий, в которых есть координаты pointer_x/pointer_y.
        У ботов часто 0 (нет мыши на уровне событий).
    pointer_x/y_mean/std/min/max: статистики по координатам внутри событий с курсором.
    share_ptr_zero: доля событий с курсором ровно в (0, 0).
    share_ptr_center: доля событий с курсором в центре экрана (типичное значение для ботов).
    ptr_std_xy_ratio: отношение std_x / std_y. У людей обычно ~1.5-2 (шире по X),
        у ботов часто отличается.
    """
    g = ev.groupby("cookie_id")
    share_ptr = g.apply(
        lambda d: float(d.pointer_x.notna().mean()),
        include_groups=False,
    ).rename("share_with_ptr").reset_index()
    ptr = ev.dropna(subset=["pointer_x", "pointer_y"])
    if len(ptr):
        stats = ptr.groupby("cookie_id")[["pointer_x", "pointer_y"]].agg(["mean", "std", "min", "max"])
        stats.columns = [a + "_" + b for a, b in stats.columns]
        stats = stats.reset_index()
    else:
        stats = pd.DataFrame({"cookie_id": meta.cookie_id})
    bad = ev.assign(
        is_zero=lambda d: ((d.pointer_x.fillna(-1) == 0) & (d.pointer_y.fillna(-1) == 0)).astype(int),
        is_center=lambda d: (
            (d.pointer_x.fillna(-1).between(660, 680))
            & (d.pointer_y.fillna(-1).between(360, 380))
        ).astype(int),
    )
    bad_share = bad.groupby("cookie_id")[["is_zero", "is_center"]].mean().reset_index()
    bad_share.columns = ["cookie_id", "share_ptr_zero", "share_ptr_center"]
    f = meta[["cookie_id"]].merge(share_ptr, on="cookie_id", how="left")
    f = f.merge(stats, on="cookie_id", how="left")
    f = f.merge(bad_share, on="cookie_id", how="left")
    f = f.fillna(0)
    f["ptr_std_xy_ratio"] = f.get("pointer_x_std", 0) / (f.get("pointer_y_std", 0) + 1e-3)
    return f


def ua_feats(ev, meta):
    """Фичи по User-Agent.

    Для каждой куки берём самый частый UA и извлекаем:
    - ua_len_mean: длина строки UA
    - ua_bot_pat: 1 если в UA есть паттерны автоматизации (headless/selenium/curl/...)
    - ua_chrome/firefox/safari/ya: флаги семейства браузера
    - ua_mobile: 1 если в UA есть признак мобильного устройства
    - ua_headless: 1 если это HeadlessChrome
    Также считаем share_ua_bot_pat: доля событий куки с ботовым паттерном в UA.
    """
    g = ev.groupby("cookie_id")
    n_ua = g.user_agent.nunique().rename("n_unique_ua").reset_index()

    # самый частый UA у куки
    top_ua = (
        ev.groupby(["cookie_id", "user_agent"]).size()
        .reset_index(name="n")
        .sort_values(["cookie_id", "n"], ascending=[True, False])
        .drop_duplicates("cookie_id")[["cookie_id", "user_agent"]]
    )
    s = top_ua.user_agent.fillna("")
    ua_df = pd.DataFrame({
        "ua_len_mean": s.str.len(),
        "ua_bot_pat":  s.str.contains(UA_BOT_RE).astype(int),
        "ua_chrome":   s.str.contains("Chrome/",  regex=False).astype(int),
        "ua_firefox":  s.str.contains("Firefox/", regex=False).astype(int),
        "ua_safari":   (s.str.contains("Safari/", regex=False)
                        & ~s.str.contains("Chrome/", regex=False)).astype(int),
        "ua_ya":       s.str.contains("YaBrowser", regex=False).astype(int),
        "ua_mobile":   s.str.contains("Mobile|Android|iPhone", regex=True).astype(int),
        "ua_headless": s.str.contains("HeadlessChrome", regex=False).astype(int),
    })
    top_ua = pd.concat([top_ua.reset_index(drop=True), ua_df], axis=1)

    ev_ua = ev.assign(bot_pat=ev.user_agent.fillna("").str.contains(UA_BOT_RE).astype(int))
    share_bot = ev_ua.groupby("cookie_id")["bot_pat"].mean().rename("share_ua_bot_pat").reset_index()

    f = meta[["cookie_id"]].merge(n_ua, on="cookie_id", how="left")
    f = f.merge(top_ua.drop(columns=["user_agent"]), on="cookie_id", how="left")
    f = f.merge(share_bot, on="cookie_id", how="left")
    return f.fillna(0)


def meta_feats(meta):
    """Метаинформация о куки.

    cookie_age_hours/days/log: сколько времени прошло от создания куки до начала окна.
        У ботов-однодневок это значение близко к 0.
    window_dow/hour/day: временные метки начала окна.
    """
    age = (meta.window_start_ts - meta.cookie_created_at).dt.total_seconds()
    return pd.DataFrame({
        "cookie_id":         meta.cookie_id,
        "cookie_age_hours":  age / 3600.0,
        "cookie_age_days":   age / 86400.0,
        "cookie_age_log":    np.log1p(age / 3600.0),
        "window_dow":        meta.window_start_ts.dt.dayofweek,
        "window_hour":       meta.window_start_ts.dt.hour,
        "window_day":        meta.window_start_ts.dt.day,
    })


def behavior_feats(ev, meta):
    """Поведенческие фичи (captcha, login, item-reuse, transitions).

    Капча: бот провоцирует капчу чаще. Считаем:
    - has_captcha: была ли капча у куки
    - cnt_captcha_v: сколько раз показалась капча
    - share_after_captcha: доля событий ПОСЛЕ первой капчи (бот продолжает работать)

    Логин: аналогично has_login, cnt_login_v.

    Item-reuse: бот может просматривать одни и те же объявления много раз.
    - item_reuse_share: доля item_id, которые встречались больше одного раза
    - item_max_repeat: максимальное число повторов одного item_id

    Скорость: доля интервалов короче 1/5/30 секунд (боты часто кликают быстрее людей).

    Платформа: std индекса платформы (если 1 платформа - 0, если много - большое значение).

    Первый event: бот часто начинает с поиска, человек может с чего угодно.
    - is_first_search: первое событие = search_results_view

    Биграммы событий: топ-1 пара (prev, curr) и её доля. Боты часто имеют
    однотипные последовательности (например, всегда search -> item_view).
    - top_pair_share: доля самой частой пары
    """
    g = ev.groupby("cookie_id")
    n = g.size().rename("n_events")
    has_captcha = g.apply(lambda d: int((d.event_name == "captcha_shown").any()), include_groups=False).rename("has_captcha").reset_index()
    cnt_captcha = g.apply(lambda d: int((d.event_name == "captcha_shown").sum()), include_groups=False).rename("cnt_captcha_v").reset_index()
    has_login   = g.apply(lambda d: int((d.event_name == "login").any()),        include_groups=False).rename("has_login").reset_index()
    cnt_login   = g.apply(lambda d: int((d.event_name == "login").sum()),        include_groups=False).rename("cnt_login_v").reset_index()

    # доля событий после первой капчи
    ev = ev.sort_values(["cookie_id", "event_ts"]).reset_index(drop=True)
    ev["is_captcha"] = (ev.event_name == "captcha_shown").astype(int)
    ev["row"] = ev.groupby("cookie_id").cumcount()
    first_cap = (
        ev[ev.is_captcha == 1].groupby("cookie_id")["row"].min()
        .rename("first_cap").reset_index()
    )
    ev2 = ev.merge(first_cap, on="cookie_id", how="left")
    share_after = (
        ev2.assign(after=ev2.row > ev2.first_cap)
        .groupby("cookie_id")["after"].mean()
        .rename("share_after_captcha").reset_index()
    )

    # повторы item_id
    g_item = ev.groupby(["cookie_id", "item_id"]).size().reset_index(name="c")
    item_reuse = (
        g_item.assign(rep=(g_item.c > 1).astype(int))
        .groupby("cookie_id")["rep"].mean()
        .rename("item_reuse_share").reset_index()
    )
    max_repeat = g_item.groupby("cookie_id")["c"].max().rename("item_max_repeat").reset_index()

    # доля коротких интервалов
    ev["dt"] = ev.groupby("cookie_id")["event_ts"].diff().dt.total_seconds()

    def frac_lt(s, t):
        s = s.dropna()
        return float((s < t).mean()) if len(s) else 0.0

    fast = ev.groupby("cookie_id")["dt"].agg(
        dt_frac_lt_1s  =lambda s: frac_lt(s, 1),
        dt_frac_lt_5s  =lambda s: frac_lt(s, 5),
        dt_frac_lt_30s =lambda s: frac_lt(s, 30),
    ).reset_index()

    # std индекса платформы (1 платформа -> 0; несколько -> большое)
    plat_idx = pd.Series({p: i for i, p in enumerate(PLATFORMS)})
    plat_norm = ev.platform.astype(str).str.lower().map(plat_idx).fillna(-1).astype(float)
    plat_std = ev.assign(_p=plat_norm).groupby("cookie_id")["_p"].std().rename("plat_index_std").reset_index()
    plat_n = ev.groupby("cookie_id").platform.apply(lambda s: s.str.lower().nunique()).rename("plat_norm_n").reset_index()

    # первый event - поиск?
    first = ev.groupby("cookie_id").first()
    is_first_search = (first.event_name == "search_results_view").astype(int).rename("is_first_search").reset_index()

    # самая частая биграмма (prev_event, event_name) и её доля
    ev["prev"] = ev.groupby("cookie_id")["event_name"].shift(1)
    pairs = ev.dropna(subset=["prev"]).groupby(
        ["cookie_id", "prev", "event_name"]
    ).size().reset_index(name="c")

    def top_share(grp):
        return float(grp.c.max() / grp.c.sum()) if len(grp) else 0.0

    top_pair = pairs.groupby("cookie_id").apply(top_share, include_groups=False).rename("top_pair_share").reset_index()

    f = meta[["cookie_id"]].merge(
        pd.DataFrame({"cookie_id": n.index, "n_events": n.values}),
        on="cookie_id", how="left",
    )
    for part in [has_captcha, cnt_captcha, has_login, cnt_login, share_after,
                 item_reuse, max_repeat, fast, plat_std, plat_n, is_first_search, top_pair]:
        f = f.merge(part, on="cookie_id", how="left")
    return f.fillna(0)


def bot_flags(X, meta):
    """Ручные эвристические флаги «похож на бота».

    Каждый флаг - это простое правило на уже посчитанных фичах.
    Деревья их бы и сами нашли, но явные флаги помогают модели и делают
    поведение более интерпретируемым.

    Используем safe_get вместо X.get(col, default), потому что default -
    скаляр, а нам нужен Series нужной длины для булевых операций.
    """
    n = len(meta)

    def safe_get(col, default=0):
        if col in X.columns:
            return X[col]
        return pd.Series([default] * n, index=X.index[:n] if len(X) else None)

    return pd.DataFrame({
        "cookie_id": meta.cookie_id.values,
        # в окне 0-1 событий (похоже на пинг/headless-тест)
        "flag_single_event":      (safe_get("n_events") <= 1).astype(int),
        # курсор нигде не появлялся
        "flag_no_pointer":        (safe_get("share_with_ptr") == 0).astype(int),
        # больше 30% интервалов попали в коридор +-5% от своей медианы
        "flag_very_periodic":     (safe_get("dt_periodic5") > 0.3).astype(int),
        # самый частый UA содержит ботовый паттерн
        "flag_ua_bot":            (safe_get("ua_bot_pat") == 1).astype(int),
        # больше половины событий с ботовым UA
        "flag_share_ua_bot_high": (safe_get("share_ua_bot_pat") >= 0.5).astype(int),
        # нетипично много событий за сутки
        "flag_many_events":       (safe_get("n_events") > 50).astype(int),
        # средняя страница поиска >= 5 (бот листает глубоко)
        "flag_high_page":         (safe_get("page_mean") >= 5).astype(int),
        # вообще не было запросов
        "flag_no_queries":        (safe_get("uniq_query") == 0).astype(int),
        # вся активность уложилась в <60 секунд
        "flag_short_session":     ((safe_get("session_span") > 0) & (safe_get("session_span") < 60)).astype(int),
        # активность почти весь день
        "flag_long_session":      (safe_get("session_span") > 80000).astype(int),
        # один UA на все события
        "flag_single_ua":         (safe_get("n_unique_ua", 1) == 1).astype(int),
        # слишком мало событий для расчёта dt
        "flag_no_dt":             (safe_get("n_dt") < 3).astype(int),
        # суммарный "подозрительный" счёт (0-4)
        "flag_suspicious_combo": (
            (safe_get("share_with_ptr") == 0).astype(int)
            + (safe_get("ua_bot_pat") == 1).astype(int)
            + (safe_get("dt_periodic5") > 0.3).astype(int)
            + (safe_get("n_events") > 50).astype(int)
        ),
    })


def cross_cookie_feats(ev_tr, ev_te, meta_tr, X_all):
    """Frequency-only кросс-cookie фичи по user_agent и item_id.

    Считаем частоту встречаемости UA и item_id во всех событиях (train+test).
    Это НЕ утечка: мы не используем train-лейблы, только сам факт частотности.

    Важно: ранние версии этой функции возвращали ua_bot_rate и item_bot_rate_smooth
    (доля ботов по UA/item_id), которые давали утечку через train-лейблы.
    Теперь они УБРАНЫ. Остались только frequency-фичи.

    Колонки:
        ua_count: сколько кук в train+test используют этот UA
        ua_log_count: log1p(ua_count)
        item_count: сколько раз item_id встречается во всех событиях
        item_log_count: log1p(item_count)

    Идея: популярные UA / item_id у ботов — это частый паттерн. Даже без лейблов,
    частотность коррелирует с ботоводством.
    """
    ev_all = pd.concat([ev_tr, ev_te], ignore_index=True)

    # --- по user_agent: считаем сколько УНИКАЛЬНЫХ кук используют каждый UA
    # Берём самую частую пару cookie_id -> user_agent
    top_ua = (
        ev_all.groupby(["cookie_id", "user_agent"]).size()
        .reset_index(name="n")
        .sort_values(["cookie_id", "n"], ascending=[True, False])
        .drop_duplicates("cookie_id")[["cookie_id", "user_agent"]]
    )
    ua_count_df = (
        top_ua.groupby("user_agent").size()
        .rename("ua_count").reset_index()
    )
    ua_count_df["ua_log_count"] = np.log1p(ua_count_df["ua_count"])

    # --- по item_id: считаем частоту item_id во всех событиях
    item_count_df = (
        ev_all.dropna(subset=["item_id"])
        .groupby("item_id").size()
        .rename("item_count").reset_index()
    )
    item_count_df["item_log_count"] = np.log1p(item_count_df["item_count"])

    # --- маппинг cookie -> top UA и top item ---
    top_item = (
        ev_all.sort_values(["cookie_id", "event_ts"])
        .dropna(subset=["item_id"])
        .drop_duplicates("cookie_id", keep="first")[["cookie_id", "item_id"]]
    )

    X2 = X_all.merge(top_ua[["cookie_id", "user_agent"]], on="cookie_id", how="left")
    X2 = X2.merge(ua_count_df, on="user_agent", how="left")
    X2 = X2.merge(top_item, on="cookie_id", how="left")
    X2 = X2.merge(item_count_df, on="item_id", how="left")

    # заполняем NaN нулями (куки без UA или item_id)
    for c in ["ua_count", "ua_log_count", "item_count", "item_log_count"]:
        X2[c] = X2[c].fillna(0)

    return X2[["cookie_id", "ua_count", "ua_log_count", "item_count", "item_log_count"]]


def build_xy(ev_tr, ev_te, train, test):
    """Собирает X_train, X_test и y_train.

    Шаги:
    1. Для каждой группы фич (basic/timing/event_mix/platform/search/diversity/pointer/ua/meta/behavior)
       считаем их на train и test отдельно.
    2. Добавляем bot_flags поверх объединённых фич.
    3. Добавляем 4 frequency-only cross-cookie фичи (ua_count, ua_log_count,
       item_count, item_log_count) — считаются по train+test без train-лейблов.
    4. Выравниваем набор колонок train и test (могут различаться из-за разного
       разнообразия категорий/UA), fillna(0).

    Возвращает: Xtr, Xte, ytr (в том же порядке cookie_id что и в meta).
    """
    def merge_all(ev, meta):
        f = basic_feats(ev, meta)
        f = f.merge(timing_feats(ev, meta), on="cookie_id", how="left")
        f = f.merge(event_mix_feats(ev, meta), on="cookie_id", how="left")
        f = f.merge(platform_feats(ev, meta), on="cookie_id", how="left")
        f = f.merge(search_feats(ev, meta), on="cookie_id", how="left")
        f = f.merge(diversity_feats(ev, meta), on="cookie_id", how="left")
        f = f.merge(pointer_feats(ev, meta), on="cookie_id", how="left")
        f = f.merge(ua_feats(ev, meta), on="cookie_id", how="left")
        f = f.merge(meta_feats(meta), on="cookie_id", how="left")
        f = f.merge(behavior_feats(ev, meta), on="cookie_id", how="left")
        f = f.merge(bot_flags(f, meta), on="cookie_id", how="left")
        return f

    Xtr = merge_all(ev_tr, train)
    Xte = merge_all(ev_te, test)
    cc = cross_cookie_feats(ev_tr, ev_te, train, pd.concat([Xtr, Xte], ignore_index=True))
    Xtr = Xtr.merge(cc, on="cookie_id", how="left")
    Xte = Xte.merge(cc, on="cookie_id", how="left")

    common = [c for c in Xtr.columns if c in Xte.columns and c != "cookie_id"]
    Xtr = Xtr[["cookie_id"] + common].fillna(0)
    Xte = Xte[["cookie_id"] + common].fillna(0)
    ytr = train.set_index("cookie_id").loc[Xtr.cookie_id, "target"]
    return Xtr, Xte, ytr
