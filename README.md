# Avito anti-bot: детекция парсеров

Решение задачи бинарной классификации cookie_id на ботов и людей по 24-часовому окну событий.

## Задача

По событиям внутри суточного окна наблюдения построить модель, выдающую каждой `cookie_id` оценку от 0 до 1: чем выше, тем вероятнее принадлежность к положительному классу (бот). Подробное описание в `TASK.md`.

**Метрика:** `max Precision` при условии `Recall >= 0.70` (рассчитывается по группам одинакового `score`, поэтому округлять `score` нельзя).

## Структура проекта

```
.
├── data/                          # данные (train/test/events) - в .gitignore
├── notebooks/
│   ├── 01_eda.ipynb               # разведка данных
│   ├── 02_experiments.ipynb       # история экспериментов с фичами
│   └── 03_final_solution.ipynb    # финальное решение (главный)
├── output/
│   └── submission.csv             # предсказания (пересоздаётся запуском 03)
├── features.py                    # все функции построения фич
├── metric.py                      # официальная метрика (из условия)
├── requirements.txt
├── sample_submission.csv          # формат для проверки submission.csv
├── TASK.md                        # условие задачи (исходное)
└── README.md                      # этот файл
```

## Что внутри

- `features.py` — функции построения ~165 фич из событий: тайминг, разнообразие, распределение по типам событий и платформам, поиск, pointer, user-agent, мета-cookie, поведение (captcha/login), bot-флаги, кросс-cookie по UA и item_id.
- `01_eda.ipynb` — разведка: что в данных, как отличаются боты от людей.
- `02_experiments.ipynb` — пошаговые эксперименты: какие группы фич дают какой прирост.
- `03_final_solution.ipynb` — финальное решение: фичи → три модели (LightGBM, CatBoost, XGBoost) → ансамбль через rank-mean → `output/submission.csv`.

## Как запустить

1. Положить файлы в `data/`:
   - `data/train.csv`
   - `data/test.csv`
   - `data/events.csv.gz`
2. Запустить `notebooks/03_final_solution.ipynb` (по порядку, все ячейки).
3. Готовый файл появится в `output/submission.csv`.

Точка входа для проверки - `notebooks/03_final_solution.ipynb`. Он автономный, при запуске собирает фичи, обучает три модели, делает ансамбль и сохраняет `submission.csv`.

## Окружение

- Python 3.13
- Зависимости в `requirements.txt`: numpy, pandas, scipy, scikit-learn, lightgbm, catboost, xgboost.

Установка:
```
conda create -n avito-bot python=3.13
conda activate avito-bot
pip install -r requirements.txt
```

## Подход (коротко)

Гибрид эвристик и градиентного бустинга.

1. **Фильтрация событий по окну:** строго `[window_start, window_end)`.
2. **~165 фич по cookie_id:**
   - базовые счётчики
   - тайминг (dt-статистики, периодичность, max-burst, активные часы)
   - распределение по типам событий и платформам
   - поиск (запросы, страницы)
   - pointer (доля и std координат)
   - user-agent (длина, паттерны headless/selenium/curl, семейство браузера)
   - мета-cookie (возраст, день недели)
   - поведение (captcha, login, item-reuse, биграммы событий)
   - bot-флаги (ручные булевы признаки)
   - **кросс-cookie** (частоты и сглаженный bot-rate по user_agent и item_id по train)
3. **Три модели:** LightGBM, CatBoost, XGBoost на одних и тех же фичах.
4. **Ансамбль:** rank-mean, веса подбираются перебором на time-val.
5. **Валидация:** time-based split (последние ~3 дня train), метрика — `precision_at_recall` из `metric.py`.
6. **Submission:** `output/submission.csv` с проверкой формата.

## Результаты

На time-val (последние ~3 дня train, 1 951 кука, 8.2% ботов):

| Модель | P@R0.7 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| LightGBM | 0.9496 | 0.9919 | 0.9173 |
| CatBoost | 0.9280 | 0.9923 | 0.9232 |
| XGBoost | 0.9524 | 0.9920 | 0.9191 |
| Ансамбль (auto-weight на val) | ~0.95 | - | - |

Главный сигнал в датасете — **кросс-cookie**: если UA или объявление этой куки уже встречались у ботов в train, она почти наверняка тоже бот (item_bot_rate_smooth и ua_bot_rate_smooth — топ-фичи по важности).

## Воспроизводимость

- `random_state=42` зафиксирован в ноутбуках.
- Все версии библиотек в `requirements.txt`.
- При повторном запуске `03_final_solution.ipynb` получается тот же `submission.csv`.

## Что пробовали и не зашло

- **Multi-seed averaging** (5 сидов): не дало прироста.
- **CatBoost с cat_features**: просел до 0.94.
- **CatBoost с iterations=1200/lr=0.04**: не лучше дефолта.

## Что можно ещё попробовать

- Optuna-тюнинг (depth, l2, leaf_size) для каждой модели.
- Target encoding по токенам UA, а не по целой строке.
- NN поверх последовательности событий (LSTM/Transformer).
- Расширить bot-флаги (например, медиана времени search → item_view).

## Ограничения

- Зависит от того, что боты в test используют те же UA/item_id, что и боты в train. Если придут совсем новые парсеры, кросс-cookie фичи потеряют силу.
- Модели не покрывают новые типы автоматизации, которые появятся в будущем.
