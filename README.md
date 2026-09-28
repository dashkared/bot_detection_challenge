# Avito Anti-Bot: детекция парсеров по 24-часовому окну событий

Решение задачи бинарной классификации `cookie_id` на ботов и людей.
Финальный скор: **P@R0.7 = 0.82188** на скрытом тесте.

## Структура проекта

```
.
├── data/                          # данные (train/test/events) — в .gitignore
├── notebooks/
│   ├── 01_eda.ipynb               # разведка данных
│   ├── 02_experiments.ipynb       # история экспериментов с фичами
│   └── 03_final_solution.ipynb          # финальное решение (главный)
├── output/
│   └── submission.csv              # предсказания (пересоздаётся запуском 03)
├── features.py                     # функции построения 162 фич
├── metric.py                       # официальная метрика (P@R0.7)
├── requirements.txt
├── sample_submission.csv           # формат submission
├── TASK.md                         # условие задачи
└── README.md                       # этот файл
```

## Быстрый старт

1. Положить файлы в `data/`:
   - `data/train.csv`
   - `data/test.csv`
   - `data/events.csv.gz`
2. Запустить `notebooks/03_final_solution.ipynb` (все ячейки по порядку).
3. Готовый файл появится в `output/submission.csv`.

## Данные

- **train:** 11 091 cookie (06–19 апреля), 8.1% боты
- **test:** 4 909 cookie (20–26 апреля)
- **events.csv.gz:** события внутри окон наблюдения

## Подход

### 1. Построение фич (162 штуки)

Фильтрация событий строго по `[window_start, window_end)`.
Группы фич (все вычисляются только по событиям внутри окна):

| Группа | Что считаем | Примеры |
|---|---|---|
| **basic_feats** | размер сессии, разнообразие сущностей | n_events, n_unique_item/cat/loc/seller |
| **timing_feats** | интервалы между событиями, периодичность, активные часы | dt_mean/std/median/q95, dt_periodic5/15, max_burst_60s, hour_entropy, share_night |
| **event_mix_feats** | one-hot по типам событий + доля | cnt/share_search_results_view, cnt/share_item_view, engagement_per_view |
| **platform_feats** | распределение по платформам | plat_web/android/ios/desktop, plat_share_*, n_platforms |
| **search_feats** | поведение в поиске | n_search, uniq_queries, query_len_mean/max, page_mean/max |
| **diversity_feats** | разнообразие контента, энтропия | uniq_item/cat/loc/seller, cat_entropy |
| **pointer_feats** | координаты курсора | share_with_ptr, pointer_x/y_std/mean, share_ptr_zero/center |
| **ua_feats** | User-Agent | ua_len, ua_bot_pat, ua_chrome/firefox/safari, ua_mobile |
| **meta_feats** | возраст и время куки | cookie_age_hours, window_dow/hour/day |
| **behavior_feats** | капча, логин, повторы item, биграммы | has_captcha, item_reuse_share, top_pair_share, dt_frac_lt_1s |
| **bot_flags** | ручные эвристические флаги | flag_no_pointer, flag_very_periodic, flag_ua_bot, flag_short_session |

### 2. Модели и валидация

- **Валидация:** 5-fold StratifiedKFold (random_state=42) — честнее time-based split
- **Модели:** LightGBM + CatBoost (ансамбль через rank-mean)
- **Метрика на CV:** P@R0.7 (precision_at_recall из metric.py)

### 3. Финальный ансамбль

```
score = w_l * rank01(LGBM_proba) + (1 - w_l) * rank01(CatBoost_proba)
w_l подбирается перебором на CV
```

### 4. Почему НЕ кросс-cookie с лейблами

Изначальная версия использовала `ua_bot_rate_smooth` и `item_bot_rate_smooth`
(сглаженный bot-rate по UA/item_id из train). На time-val: **P@R0.7 = 0.95**.
На скрытом тесте: **0.41** — катастрофический провал.

**Причина:** test куки содержат UA/item_id, не виденные в train. Фичи дефолтят
к глобальной средней и теряют предсказательную силу. Это **утечка через будущее**:
фичи зависят от всей выборки, а не только от данных доступных на момент предсказания.

**Решение:** убраны все cross-cookie фичи с использованием train-лейблов.
Оставлены только внутри-куковые фичи + 4 frequency cross-cookie (162 штуки).

## Результаты

### Честная кросс-валидация (5-fold StratifiedKFold)

| Модель | P@R0.7 (CV) |
|---|---:|
| LightGBM | ~0.80 |
| CatBoost | ~0.79 |
| Ансамбль (rank-mean) | ~0.80 |

### Скрытый тест

**Финальный submission: P@R0.7 = 0.82188**

Разрыв CV=0.80 → hidden test=0.82 может объясняться временны́м сдвигом
в паттернах ботов между 06–19 и 20–26 апреля (тестовые боты могут быть
более «классическими» и лучше детектируются внутри-куковыми фичами).

## Конфигурация моделей

**LightGBM:**
```python
LGBMClassifier(
    n_estimators=1000, learning_rate=0.02,
    num_leaves=31, min_child_samples=40,
    subsample=0.65, colsample_bytree=0.45,
    reg_lambda=10.0, max_depth=4
)
```

**CatBoost:**
```python
CatBoostClassifier(
    iterations=600, learning_rate=0.04,
    depth=4, l2_leaf_reg=6.0
)
```

## Что пробовали и не зашло

- **Кросс-cookie фичи с train-лейблами** (ua_bot_rate_smooth, item_bot_rate_smooth):
  time-val=0.95, hidden test=**0.41** — утечка.
- **9 item-level фич** (item_once, item_twice, item_thrice_plus, item_view_max/mean/std):
  CV просел с 0.80 → 0.7957, на скрытом тесте **0.80**.
- **XGBoost в ансамбле:** просел качество, в финальном решении не используется.
- **Hyperparameter tuning** (depth, reg, num_leaves): просело, оставлен дефолт.
- **Pseudo-labeling:** CV просел с 0.80 → 0.79, не применялся.

## Воспроизводимость

- `random_state=42` зафиксирован в ноутбуке.
- Python 3.13, зависимости в `requirements.txt`.
- При повторном запуске `03_final_solution.ipynb` получается тот же `submission.csv`.

## Окружение

```
conda create -n avito-bot python=3.13
conda activate avito-bot
pip install -r requirements.txt
```

## Что ещё можно попробовать

- Target encoding по токенам UA (не по всей строке).
- Фичи на последовательности событий (LSTM/Transformer) — требует больше данных.
- Анализ временно́го дрейфа: адаптивная калибровка на последние дни train.
- Optuna-тюнинг для CatBoost отдельно.
