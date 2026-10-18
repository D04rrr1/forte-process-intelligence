# Factory Digital Twin

Прототип цифрового двойника автомобильного завода для QOSTANAI AI INDUSTRY HACKATHON 2026.

## Что делает проект

Система объединяет производственный мониторинг и AI-анализ оборудования:

- мониторинг plan / fact;
- загрузка производственных линий;
- контроль качества;
- учет простоев;
- Digital Twin производственного потока;
- AI #1 — прогноз вероятности отказа;
- AI #2 — диагностика вероятной причины;
- AI #3 — подбор более безопасных параметров работы;
- рекомендации инженеру с ручным подтверждением.

## Архитектура

```text
Factory data / Machine sensors
            ↓
      Digital Twin
            ↓
     AI #1 Predict
            ↓
    AI #2 Diagnose
            ↓
    AI #3 Optimize
            ↓
     Engineer / Manager
```

## Структура проекта

```text
factory-digital-twin/
├── app.py
├── factory_data.py
├── failureai.py
├── failurecause.py
├── failurdesicion.py
├── failure_binary_model.joblib
├── failure_type_model.joblib
├── ai4i2020.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## AI Pipeline

### AI #1 — Predict
Файл: `failureai.py`

Вход:
- machine type;
- air temperature;
- process temperature;
- RPM;
- torque;
- tool wear.

Выход:
- вероятность отказа оборудования.

### AI #2 — Diagnose
Файл: `failurecause.py`

Определяет вероятные типы отказа:
- TWF — Tool Wear Failure;
- HDF — Heat Dissipation Failure;
- PWF — Power Failure;
- OSF — Overstrain Failure.

### AI #3 — Optimize
Файл: `failurdesicion.py`

Использует результаты AI #1 и AI #2 и подбирает рекомендуемые:
- RPM;
- Torque;
- Process temperature.

AI #3 не управляет оборудованием напрямую. Это decision-support слой: AI рекомендует, инженер подтверждает.

## Dataset

Для обучения predictive maintenance моделей используется AI4I 2020 Predictive Maintenance Dataset.

Файл:
`ai4i2020.csv`

Примечание: тестовые данные индустриального кейса могут иметь ограничения на публичное распространение. Перед публикацией репозитория убедитесь, что у команды есть право размещать их в открытом доступе.

## Запуск

Установить зависимости:

```bash
python3 -m pip install -r requirements.txt
```

Запустить Streamlit:

```bash
python3 -m streamlit run app.py
```

После запуска приложение будет доступно по адресу, который покажет Streamlit, обычно:

```text
http://localhost:8501
```

## Основные экраны

### Overview
- производительность;
- загрузка;
- defect rate;
- downtime;
- alerts;
- plan vs actual;
- качество по участкам.

### Digital Twin
Производственный поток:

```text
Components → Welding → Painting → Assembly → Quality Control → Finished Goods
```

### AI Control Center
Сценарий:

```text
Predict → Diagnose → Optimize
```

Показывает:
- текущий failure risk;
- risk level;
- вероятную причину;
- рекомендуемые параметры;
- optimized risk;
- ручное подтверждение рекомендации.

## Production integration

Для реального внедрения вместо тестового источника данных можно подключить:

```text
PLC / Sensors / SCADA / MES
        ↓
OPC UA / MQTT / API
        ↓
Backend
        ↓
Digital Twin + AI
```

## Технологии

- Python
- Streamlit
- Pandas
- scikit-learn
- Random Forest
- Joblib
- FastAPI
- Pydantic
- Altair

## Ограничения MVP

- используется тестовый набор данных;
- нет прямого подключения к PLC;
- рекомендации AI не отправляются на оборудование автоматически;
- реальные безопасные operating limits должны быть согласованы с инженерами завода;
- бизнес-эффект требует проверки на пилоте.

## Hackathon

Case #2: Digital Twin for automotive production

QOSTANAI AI INDUSTRY HACKATHON 2026

Industrial partner: Allur
