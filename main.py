import gradio as gr
import spaces
import torch
import os
import numpy as np
import asyncio
import logging
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Освобождение кэша GPU для оптимизации памяти
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Параметры обработки изображений
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

# Словарь для распределения слоев модели на разные устройства (CPU/GPU)
DEVICE_MAP = {
    'model.embed_tokens': 0,
    **{f'model.layers.{i}': 0 for i in range(27)},
    **{f'model.layers.{i}': 1 for i in range(27, 80)},
    'model.norm': 1,
    'visual.patch_embed': 1,
    **{f'visual.blocks.{i}': 0 for i in range(9)},
    **{f'visual.blocks.{i}': 1 for i in range(9, 32)},
    'visual.merger': 1,
    'lm_head': 0,
}

# Параметры запуска модели
MODEL_NAME = "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4"
_model_processor_cache = {}

# Описание модели
DESCRIPTION = "[Qwen2-VL-72B Demo](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4)"

# Шаблоны для сообщений
USER_PROMPT = '<|user|>\n'
ASSISTANT_PROMPT = '<|assistant|>\n'
PROMPT_SUFFIX = "<|end|>\n"

# Создание модели и процессора с кэшированием
def initialize_model_and_processor(model_name=MODEL_NAME):
    try:
        logging.info(f"Инициализация модели и процессора для: {model_name}")
        # Проверка кэша
        if model_name not in _model_processor_cache:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype="auto", device_map=DEVICE_MAP
            ).cuda().eval()

            processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
            )
            _model_processor_cache[model_name] = (model, processor)

        return _model_processor_cache[model_name]

    except Exception as e:
        logging.error(f"Ошибка инициализации модели или процессора: {e}")
        raise

# Преобразование массива в изображение и сохранение на диск
def convert_array_to_image(image_array):
    try:
        # Преобразование массива numpy в изображение PIL
        image = Image.fromarray(np.uint8(image_array))
        # Генерация уникального имени файла
        unique_filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        # Сохранение изображения
        image.save(unique_filename)
        logging.info(f"Изображение сохранено: {unique_filename}")
        return os.path.abspath(unique_filename)
    except Exception as e:
        logging.error(f"Ошибка при преобразовании массива в изображение: {e}")
        raise

def cleanup_image(image_path):
    try:
        os.remove(image_path)
        logging.info(f"Временный файл удален: {image_path}")
    except OSError as e:
        logging.error(f"Ошибка при удалении временного файла {image_path}: {e}")

# Асинхронная обработка изображений и текста
@spaces.GPU
async def process_image_and_text(image, text_input=None, model_id=MODEL_NAME):
    image_path = None
    try:
        # Преобразование и сохранение изображения
        image_path = convert_array_to_image(image)
        logging.info(f"Путь к изображению: {image_path}")

        # Загрузка модели и процессора
        model, processor = initialize_model_and_processor(model_id)

        # Формирование сообщения пользователя
        user_message = f"{USER_PROMPT}<|image_1|>\n{text_input}{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_input},
                ],
            }
        ]
        
        # Подготовка данных для инференса
        processed_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # Подготовка входных данных для модели
        inputs = processor(
            text=[processed_text],
            truncation=True,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        
        # Генерация текста на основе входных данных
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        
        # Декодирование сгенерированного текста
        output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return output_text[0]

    except Exception as e:
        logging.error(f"Ошибка при обработке изображения и текста: {e}")
        return "Произошла ошибка при обработке запроса."
    
    finally:
        # Удаление временного файла
        if image_path:
            cleanup_image(image_path)

# Создание веб-интерфейса с использованием Gradio
def create_gradio_interface():
    css_styles = """
      #output {
        height: 500px; 
        overflow: auto; 
        border: 1px solid #ccc; 
        padding: 10px;
        font-family: Arial, sans-serif;
        background-color: #f9f9f9;
        margin: 20px;
      }
      .gr-tab {
        background-color: #f1f1f1;
        padding: 15px;
        border-radius: 5px;
      }
      .gr-button {
        font-size: 16px;
        padding: 10px 20px;
      }
    """

    # Определение интерфейса с использованием Blocks
    with gr.Blocks(css=css_styles) as demo_interface:
        gr.Markdown(DESCRIPTION)
        with gr.Tab(label="Qwen2-VL-72B Input"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(label="Input Picture")
                    model_selector = gr.Dropdown(
                        choices=[MODEL_NAME], label="Model", value=MODEL_NAME
                    )
                    text_input = gr.Textbox(label="Question")
                    submit_btn = gr.Button(value="Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output Text")

            # Асинхронная обработка нажатия кнопки отправки
            submit_btn.click(process_image_and_text, [input_img, text_input, model_selector], [output_text])

    return demo_interface

if __name__ == "__main__":
    demo_interface = create_gradio_interface()
    demo_interface.queue(api_open=False)
    demo_interface.launch(debug=True, server_name="0.0.0.0")
