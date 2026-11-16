FROM python:3.12-slim

# Предотвращаем создание .pyc-файлов и включаем мгновенный вывод логов в stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Системные зависимости (нужны для работы нативных расширений и healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем зависимости для кэширования слоёв Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код и ресурсы проекта
COPY app.py answer.py config.py llm.py loader.py search.py ./
COPY veb_myau/ ./veb_myau/
COPY knowledge-base/ ./knowledge-base/

# Создаём директорию для кэша/данных индексов (если используется DATA_DIR=data)
RUN mkdir -p /app/data

EXPOSE 8000

# Проверка готовности сервиса (учитываем время на построение индекса при старте)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/stats || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]