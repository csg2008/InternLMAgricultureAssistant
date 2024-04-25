import json
import os
from typing import List
import streamlit as st

config_save_keys = ['room', 'model']
model_require_keys = ['model_name', 'model_path', 'model_type', 'embedding_path', 'vector_db']
config_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/data/config.json'

def load_config():
    """load config from config.json"""

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

            for k, v in config_dict.items():
                if k in config_save_keys:
                    st.session_state[k] = v

def save_config():
    '''save config to config.json'''

    config_dict = {}

    with open(config_path, 'w+', encoding='utf-8') as f:
        org_state = st.session_state.to_dict()
        for k in config_save_keys:
            if k in org_state:
                config_dict[k] = org_state[k]

        json.dump(config_dict, f, ensure_ascii=False, indent=4)

def check_config_empty(cfg, keys: List[str]) -> bool:
    '''Check if config is empty'''

    for k in keys:
        if k not in cfg or cfg[k] is None or cfg[k] == '':
            print('k:', k)
            return True

    return False
