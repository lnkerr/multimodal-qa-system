# Modified for the campus QA release: removed hard-coded answer rewriting.
# Modified for the campus QA release: project-specific model and module names.
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import re
import tempfile
import mimetypes
import time
import torch
import warnings
import gradio as gr
from datetime import datetime
from queue import Queue, Empty
from threading import Thread, Lock, Event
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, StoppingCriteria, StoppingCriteriaList
from model.model_vlm import MultimodalForCausalLM, VLMConfig
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings('ignore')
model_lock = Lock()
generation_stop_event = Event()
history_dir = os.path.join(os.path.dirname(__file__), 'chat_history')
sessions_dir = os.path.join(history_dir, 'sessions')
os.makedirs(history_dir, exist_ok=True)
os.makedirs(sessions_dir, exist_ok=True)


def guess_model_kind(model_dir):
    config_path = os.path.join(model_dir, 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception:
        config = {}

    text = json.dumps(config, ensure_ascii=False).lower()
    name = os.path.basename(model_dir).lower()
    if any(k in text for k in ('vlm', 'vision', 'image_special_token', 'image_token_len')) or '3v' in name:
        return 'vlm'
    return 'text'


def scan_models(base_dir):
    models = {}
    base_dir = os.path.abspath(base_dir)
    for d in sorted(os.listdir(base_dir), reverse=True):
        full_path = os.path.join(base_dir, d)
        if not os.path.isdir(full_path) or d.startswith('.') or d.startswith('_'):
            continue
        files = os.listdir(full_path)
        has_model = any(f.endswith(('.bin', '.safetensors')) for f in files) or 'model.safetensors.index.json' in files
        if has_model:
            models[d] = {'path': full_path, 'kind': guess_model_kind(full_path)}
    return models


def build_model_choices(models):
    counts = {}
    choices = {}
    for name, info in models.items():
        base_label = '多模态模型' if info['kind'] == 'vlm' else '大语言模型'
        counts[base_label] = counts.get(base_label, 0) + 1
        label = base_label if counts[base_label] == 1 else f'{base_label}{counts[base_label]}'
        choices[label] = name
    return choices


def model_kind_label(kind=None):
    kind = kind or current_model_kind
    return '多模态模型' if kind == 'vlm' else '大语言模型'


def load_model(model_info):
    global model, tokenizer, preprocess, lm_config, current_model_name, current_model_kind
    with model_lock:
        model_path = model_info['path']
        current_model_kind = model_info['kind']
        [sys.modules.pop(k) for k in list(sys.modules) if 'transformers_modules' in k]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        preprocess = None
        lm_config = model.config
        if current_model_kind == 'vlm':
            model.vision_encoder, model.processor = MultimodalForCausalLM.get_vision_model(vision_model_path)
            if model.vision_encoder is None: raise FileNotFoundError(f"视觉编码器未找到: {vision_model_path}")
            preprocess = model.processor

        model = model.half().eval().to(device)
        if current_model_kind == 'vlm':
            model.vision_encoder = model.vision_encoder.to(device)

        current_model_name = os.path.basename(model_path)
        param_str = f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M'
        print(f'已加载 {current_model_name} [{model_kind_label()}]')
        return f"已加载: {model_kind_label()} "


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


class StopOnEvent(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


def chat(prompt, current_image_path=None):
    global temperature, top_p
    pixel_values = None
    user_prompt = prompt
    if current_image_path:
        if current_model_kind != 'vlm':
            yield '当前选择的是大语言模型，不支持图片输入。请切换到多模态模型或移除图片后再试。'
            return
        image = Image.open(current_image_path).convert('RGB')
        pixel_values = {k: v.to(model.device) for k, v in MultimodalForCausalLM.image2tensor(image, preprocess).items()}
        user_prompt = f'{lm_config.image_special_token * lm_config.image_token_len}\n{prompt}'
    messages = [{"role": "user", "content": user_prompt}]

    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )[-max_seq_len + 1:]

    with torch.no_grad():
        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(device)
        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            try:
                with model_lock:
                    generate_kwargs = {
                        'max_new_tokens': max_seq_len,
                        'do_sample': True,
                        'temperature': temperature,
                        'top_p': top_p,
                        'attention_mask': inputs.attention_mask,
                        'pad_token_id': tokenizer.pad_token_id,
                        'eos_token_id': tokenizer.eos_token_id,
                        'streamer': streamer,
                        'stopping_criteria': StoppingCriteriaList([StopOnEvent(generation_stop_event)]),
                    }
                    if current_model_kind == 'vlm':
                        generate_kwargs['pixel_values'] = pixel_values
                    model.generate(inputs.input_ids, **generate_kwargs)
            finally:
                queue.put(None)

        generation_thread = Thread(target=_generate)
        generation_thread.start()

    while True:
            try:
                text = queue.get(timeout=0.1)
            except Empty:
                if generation_stop_event.is_set() and not generation_thread.is_alive():
                    break
                continue
            if text is None:
                break
            yield text


def format_perf(elapsed_s, first_token_s, output_text):
    output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False)) if output_text else 0
    decode_s = max(elapsed_s - (first_token_s or elapsed_s), 1e-6)
    tokens_per_s = output_tokens / decode_s if output_tokens else 0
    memory_text = 'CPU模式'
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        memory_text = f'峰值显存: {peak_mb:.1f} MB'

    first_token_text = f'{first_token_s:.2f}s' if first_token_s is not None else '-'
    return f'耗时: {elapsed_s:.2f}s | 首Token: {first_token_text} | 生成速度: {tokens_per_s:.2f} tokens/s | {memory_text}'


def make_session_id():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def make_session_title(text):
    title = str(text).replace('\n', ' ').strip() or '新对话'
    return title[:22] + ('...' if len(title) > 22 else '')


def session_path(session_id):
    safe_id = ''.join(c for c in str(session_id) if c.isalnum() or c in ('_', '-'))
    return os.path.join(sessions_dir, f'{safe_id}.json')


def is_image_path(value):
    if not isinstance(value, str):
        return False
    return os.path.splitext(value.split('?')[0].lower())[1] in {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}


def image_message(path):
    mime_type = mimetypes.guess_type(path)[0] or 'image/*'
    return {'path': path, 'mime_type': mime_type}


def normalize_message_content(content):
    if isinstance(content, dict):
        data = dict(content)
        path = data.get('path') or data.get('name')
        if path and is_image_path(path):
            return image_message(path)
        return data
    if isinstance(content, (tuple, list)) and len(content) == 1 and is_image_path(content[0]):
        return image_message(content[0])
    if isinstance(content, str):
        # 兼容旧会话：Gradio 图片消息曾被序列化为 "('/tmp/...jpg',)" 形式的字符串。
        match = re.fullmatch(r"\(['\"](.+?)['\"],\)", content.strip())
        if match and is_image_path(match.group(1)):
            return image_message(match.group(1))
        if is_image_path(content):
            return image_message(content)
        return content
    return str(content)


def restore_messages(messages):
    return [
        {
            'role': item.get('role', ''),
            'content': normalize_message_content(item.get('content', '')),
        }
        for item in messages
    ]


def save_session(session_id, history, title=None, perf=''):
    if not session_id:
        session_id = make_session_id()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = session_path(session_id)
    old = {}
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                old = json.load(f)
        except Exception:
            old = {}

    if not title:
        title = old.get('title')
    if not title:
        for item in history:
            if item.get('role') == 'user' and isinstance(item.get('content'), str):
                title = make_session_title(item.get('content'))
                break
    title = title or '新对话'

    data = {
        'id': session_id,
        'title': title,
        'created_at': old.get('created_at', now),
        'updated_at': now,
        'last_model_label': model_kind_label(),
        'last_performance': perf,
        'messages': [
            {
                'role': item.get('role', ''),
                'content': normalize_message_content(item.get('content', '')),
            }
            for item in history
        ],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return session_id


def load_sessions(limit=80):
    sessions = []
    for filename in os.listdir(sessions_dir):
        if not filename.endswith('.json'):
            continue
        path = os.path.join(sessions_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        sid = data.get('id') or filename[:-5]
        title = data.get('title') or '未命名对话'
        updated_at = data.get('updated_at') or ''
        data['_label'] = title
        data['_id'] = sid
        sessions.append(data)
    sessions.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    return sessions[:limit]


def session_choices():
    return [(item['_label'], item['_id']) for item in load_sessions()]


def load_session_item(session_id):
    if not session_id:
        return [], '请选择一个会话', None
    for item in load_sessions():
        if item['_id'] == session_id:
            return restore_messages(item.get('messages', [])), item.get('last_performance') or '会话已加载', item.get('_id')
    return [], '未找到该会话', None


def refresh_session_list(current_session_id=None):
    valid_ids = {item['_id'] for item in load_sessions()}
    value = current_session_id if current_session_id in valid_ids else None
    return gr.update(choices=session_choices(), value=value)


def new_chat():
    return [], '新会话', None, gr.update(value=None)


def delete_current_session(session_id):
    if not session_id:
        return [], '请先选择要删除的会话', None, refresh_session_list(None)

    path = session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
        return [], '会话已删除', None, refresh_session_list(None)
    return [], '会话文件不存在', None, refresh_session_list(None)


def stop_generation():
    generation_stop_event.set()
    return '正在停止生成...'


def save_chat_record(question, answer, image_path=None, perf=''):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        'time': timestamp,
        'model_label': model_kind_label(),
        'model_kind': current_model_kind,
        'question': question,
        'image_path': image_path,
        'answer': answer,
        'performance': perf,
    }

    jsonl_path = os.path.join(history_dir, 'chat_history.jsonl')
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    day_path = os.path.join(history_dir, f'{datetime.now().strftime("%Y-%m-%d")}.md')
    with open(day_path, 'a', encoding='utf-8') as f:
        f.write(f'\n## {timestamp}  {model_kind_label()}\n\n')
        if image_path:
            f.write(f'- 图片: `{image_path}`\n')
        if perf:
            f.write(f'- 性能: {perf}\n')
        f.write(f'\n**用户：**\n\n{question}\n\n**系统：**\n\n{answer}\n')


def export_history(history):
    if not history:
        return None

    lines = [
        '# 智能问答系统历史记录',
        '',
        f'导出时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
    ]
    role_names = {'user': '用户', 'assistant': '助手'}
    for item in history:
        role = role_names.get(item.get('role'), item.get('role', '未知'))
        content = item.get('content', '')
        if isinstance(content, dict):
            content = content.get('path') or json.dumps(content, ensure_ascii=False)
        lines.extend([f'## {role}', '', str(content), ''])

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.md', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        return f.name


def launch_gradio_server(server_name="0.0.0.0", server_port=7788):
    global temperature, top_p
    temperature = args.temperature
    top_p = args.top_p

    def respond(message, history, session_id):
        if not message or not message.get("text"):
            yield history + [{"role": "assistant", "content": "请输入问题"}], '等待有效输入', gr.update(value=None), refresh_session_list(session_id), session_id
            return
        generation_stop_event.clear()
        start_time = time.perf_counter()
        first_token_s = None
        if torch.cuda.is_available() and str(device).startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        files = message.get("files", [])
        img_path = files[0] if files else None
        text = message["text"]
        if img_path:
            history = history + [{"role": "user", "content": {"path": img_path}}, {"role": "user", "content": text}]
        else:
            history = history + [{"role": "user", "content": text}]
        response = ''
        for chunk in chat(text, img_path):
            if first_token_s is None:
                first_token_s = time.perf_counter() - start_time
            response = response + chunk
            elapsed_s = time.perf_counter() - start_time
            yield history + [{"role": "assistant", "content": response}], f'生成中... 已耗时: {elapsed_s:.2f}s', gr.update(value=None), gr.update(), session_id
        elapsed_s = time.perf_counter() - start_time
        stopped_text = ' | 已停止' if generation_stop_event.is_set() else ''
        perf = format_perf(elapsed_s, first_token_s, response) + stopped_text
        save_chat_record(text, response, img_path, perf)
        final_history = history + [{"role": "assistant", "content": response}]
        session_id = save_session(session_id, final_history, title=make_session_title(text) if not session_id else None, perf=perf)
        yield final_history, perf, gr.update(value=None), refresh_session_list(session_id), session_id

    css = "footer{display:none!important} #chatbox{border:none!important} #chatbox .label-wrap{display:none!important} #chatbox img{max-width:120px!important;max-height:120px!important;border-radius:8px} #history_panel{border-right:1px solid #e5e7eb;padding-right:12px;min-height:720px} #history_panel .wrap{border-radius:6px!important} #session_list label{font-size:0.92rem!important;line-height:1.35!important} #session_list .wrap{max-height:560px!important;overflow-y:auto!important} div.full-container{display:flex!important;flex-direction:row!important;flex-wrap:nowrap!important;align-items:center!important} div.full-container>.thumbnails{flex-shrink:0!important;max-width:50px!important} div.full-container>.input-container{flex:1!important} .input-wrapper{display:flex!important;flex-direction:row!important;flex-wrap:nowrap!important;align-items:center!important} .input-wrapper>.thumbnails{flex-shrink:0!important;max-width:50px!important} .input-wrapper>.input-row{flex:1!important} .thumbnail-image{max-width:40px!important;max-height:40px!important} textarea{overflow-y:hidden!important} #chatbox,#chatbox *{scrollbar-width:none!important;-ms-overflow-style:none!important} #chatbox::-webkit-scrollbar,#chatbox *::-webkit-scrollbar{display:none!important;width:0!important;height:0!important}"
    with gr.Blocks(title="基于多模态大模型的智能问答系统", css=css) as demo:
        session_state = gr.State(None)
        gr.HTML('<div style="display:flex;align-items:center;justify-content:center;margin:-8px 0 -6px 0"><span style="font-family:SimSun, 宋体, serif;font-size:1.5rem;font-weight:bold;font-style:normal">基于多模态大模型的智能问答系统</span></div>')
        with gr.Row():
            with gr.Column(scale=1, min_width=260, elem_id="history_panel"):
                gr.Markdown("### 对话")
                new_chat_btn = gr.Button("新聊天")
                refresh_btn = gr.Button("刷新会话")
                session_list = gr.Radio(choices=session_choices(), label="最近", elem_id="session_list", interactive=True)
                delete_btn = gr.Button("删除当前会话")
                export_btn = gr.Button("导出当前对话")
                history_file = gr.File(label="导出文件", interactive=False)
            with gr.Column(scale=5):
                try:
                    chatbot = gr.Chatbot(label="", height=560, elem_id="chatbox", type="messages", show_label=False)
                except TypeError:
                    chatbot = gr.Chatbot(label="", height=560, elem_id="chatbox", show_label=False)
                msg = gr.MultimodalTextbox(placeholder="输入问题，点击📎上传图片", show_label=False, submit_btn="发送")
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=list(model_choice_dict.keys()),
                        value=current_model_display_name,
                        show_label=False,
                        scale=1
                    )
                    perf_text = gr.Textbox(value="等待推理", show_label=False, interactive=False, scale=2)
                    stop_btn = gr.Button("停止生成", scale=0)

        def on_model_change(display_name):
            real_name = model_choice_dict[display_name]
            return load_model(model_dict[real_name])

        model_dropdown.change(on_model_change, [model_dropdown], [perf_text], show_progress="hidden")
        msg.submit(respond, [msg, chatbot, session_state], [chatbot, perf_text, msg, session_list, session_state], show_progress="hidden")
        session_list.change(load_session_item, [session_list], [chatbot, perf_text, session_state], show_progress="hidden")
        refresh_btn.click(refresh_session_list, [session_state], [session_list], show_progress="hidden")
        new_chat_btn.click(new_chat, None, [chatbot, perf_text, session_state, session_list], show_progress="hidden")
        delete_btn.click(delete_current_session, [session_state], [chatbot, perf_text, session_state, session_list], show_progress="hidden")
        stop_btn.click(stop_generation, None, [perf_text], show_progress="hidden")
        export_btn.click(export_history, [chatbot], [history_file], show_progress="hidden")
        demo.load(refresh_session_list, [session_state], [session_list], show_progress="hidden")
        demo.launch(server_name=server_name, server_port=server_port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chat with multimodal QA model")
    parser.add_argument('--load_from', default='./', type=str, help="transformers模型扫描目录")
    parser.add_argument('--vision_model', default='../model/siglip2-base-p32-256-ve', type=str, help="视觉编码器路径")
    parser.add_argument('--temperature', default=0.7, type=float, help="生成温度")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    args = parser.parse_args()

    device = args.device
    max_seq_len = args.max_seq_len
    vision_model_path = args.vision_model
    model_dict = scan_models(args.load_from)
    if not model_dict:
        print(f"未在 {os.path.abspath(args.load_from)} 找到transformers模型")
        exit(1)
    model_choice_dict = build_model_choices(model_dict)
    current_model_name = list(model_dict.keys())[0]
    current_model_display_name = next(
        display_name for display_name, real_name in model_choice_dict.items()
        if real_name == current_model_name
    )
    load_model(model_dict[current_model_name])
    launch_gradio_server(server_name="0.0.0.0", server_port=8888)


