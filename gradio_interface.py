import gradio as gr
from prompt_defender import PromptDefenderClassifier

# Инициализация классификатора
classifier = PromptDefenderClassifier()

def classify_prompt(prompt):
    """
    Классифицирует промпт как jailbreak или безопасный.
    Возвращает результат и вероятность.
    """
    try:
        # Получение результата от модели
        result = classifier.check_on_bad_request(prompt)
        return "Jailbreak" if result == 1 else "Safe"
    except Exception as e:
        return f"Ошибка: {str(e)}"

# Кастомный минималистичный дизайн
with gr.Blocks(css=".gradio-container {background-color: white; font-family: Arial; text-align: center;}") as interface:
    gr.Markdown(
        """
        <h1 style="color: black; margin-top: 2em;">Prompt Defender</h1>
        <p style="color: gray; font-size: 18px;">Введите текст и узнайте, является ли он jailbreak или безопасным.</p>
        """, 
        elem_id="header"
    )
    
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            input_text = gr.Textbox(
                lines=3,
                placeholder="Введите текст для проверки",
                label="",
                elem_id="input-box"
            )
            result_text = gr.Textbox(
                label="Результат",
                interactive=False,
                elem_id="result-box"
            )
            submit_button = gr.Button("Проверить", elem_id="submit-button")

    # Логика связывания ввода, кнопки и результата
    submit_button.click(classify_prompt, inputs=input_text, outputs=result_text)

# Запуск Gradio
if __name__ == "__main__":
    #interface.launch(server_name="0.0.0.0", server_port=7860) #для локального запуска
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)