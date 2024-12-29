from fastapi import FastAPI
from prompt_defender import (
    PromptDefenderClassifier,
)  # Имя класса и файла скорректировано

app = FastAPI()

# Инициализация классификатора при запуске приложения
classifier = PromptDefenderClassifier()


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app! Use the /check_prompt endpoint."}


@app.get("/check_prompt/")
def check_prompt(
    prompt_input: str,
):  # Исправлен параметр в соответствии с ожидаемым запросом
    try:
        # Используем метод для проверки запроса
        result = classifier.check_on_bad_request(prompt_input)
        return {"result": result, "success": True}
    except Exception as e:
        # Возврат ошибки в случае сбоя
        return {"error": str(e), "success": False}
