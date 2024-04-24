import streamlit as st
from lagent.agents.internlm2_agent import INTERPRETER_CN, META_CN, PLUGIN_CN

from pages.util.util import load_config, save_config

model_cfg = {
    'model_type': '远程',
    'model_name': 'internlm2-chat-7b',
    'model_path': 'http://127.0.0.1:23333',

    'temperature': 0.7,
    'max_tokens': 256,
    'top_p': 1,
    'frequency_penalty': 0,
    'presence_penalty': 0,

    'prompt_meta': META_CN,
    'prompt_da': INTERPRETER_CN,
    'prompt_plugin': PLUGIN_CN,

    'user': [],
    'assistant': [],
    'session_history': [],
}

if 'model' not in st.session_state:
    load_config()

if 'model' not in st.session_state:
    st.session_state['model'] = model_cfg

model_type_key = 1 if st.session_state['model']['model_type'] == '远程' else 0

st.set_page_config(page_title='书生·浦语大模型设置 - 农业助手', layout='wide')

st.title('书生·浦语大模型设置')

model_type = st.radio("模型类型", ["本地", "远程"], index=model_type_key, key="model_type", horizontal=True)
model_name = st.text_input('模型名', value=st.session_state['model']['model_name'], max_chars=None, key=None, type='default', placeholder = '请输入模型名称')
model_path = st.text_input('模型路径', value=st.session_state['model']['model_path'], max_chars=None, key=None, type='default', placeholder = '请输入模型路径')
temperature = st.text_input('温度', value=st.session_state['model']['temperature'], max_chars=None, key=None, type='default', placeholder = '请输入温度')
max_tokens = st.text_input('最大令牌数', value=st.session_state['model']['max_tokens'], max_chars=None, key=None, type='default', placeholder = '请输入最大令牌数')
top_p = st.text_input('置信度', value=st.session_state['model']['top_p'], max_chars=None, key=None, type='default', placeholder = '请输入置信度')
frequency_penalty = st.text_input('频率惩罚', value=st.session_state['model']['frequency_penalty'], max_chars=None, key=None, type='default', placeholder = '请输入频率惩罚')
presence_penalty = st.text_input('位置惩罚', value=st.session_state['model']['presence_penalty'], max_chars=None, key=None, type='default', placeholder = '请输入位置惩罚')
prompt_meta = st.text_area('系统提示词', value=st.session_state['model']['prompt_meta'], max_chars=None, key=None, placeholder = '请输入系统提示词')
prompt_plugin = st.text_area('插件提示词', value=st.session_state['model']['prompt_plugin'], max_chars=None, key=None, placeholder = '请输入插件提示词')

saved = st.button('保存设置')

if saved:
    st.session_state['model']['model_type'] = model_type
    st.session_state['model']['model_name'] = model_name
    st.session_state['model']['model_path'] = model_path
    st.session_state['model']['prompt_meta'] = prompt_meta
    st.session_state['model']['prompt_plugin'] = prompt_plugin
    st.session_state['model']['temperature'] = temperature
    st.session_state['model']['max_tokens'] = max_tokens
    st.session_state['model']['top_p'] = top_p
    st.session_state['model']['frequency_penalty'] = frequency_penalty
    st.session_state['model']['presence_penalty'] = presence_penalty

    save_config()