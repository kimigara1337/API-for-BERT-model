FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /workdir

# Копируем только requirements.txt перед установкой зависимостей
COPY requirements.txt /workdir/requirements.txt

# Устанавливаем зависимости (будет кэшироваться, если requirements.txt не изменился)
RUN pip install --no-cache-dir -r requirements.txt

# Копируем оставшиеся файлы проекта
COPY . /workdir

# Открываем порт для работы
EXPOSE 8080

# Указываем команду для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
