import streamlit as st
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


st.set_page_config(page_title='智慧大棚中心 - 农业助手', layout='wide')

st.title('🤠 智慧大棚中心')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'model' not in st.session_state:
    with st.container():
        st.warning('请先配置书生·浦语模型参数')
elif 'room' not in st.session_state or 'room' not in st.session_state['room'] or 0 == len(st.session_state['room']['room']):
    with st.container():
        st.warning('请先配置大棚参数')
else:
    with st.container():
        st.header('智慧大棚交互中心')

        for message in st.session_state['messages']:
            if isinstance(message, HumanMessage):
                with st.chat_message('user'):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message('assistant'):
                    st.markdown(message.content)
        prompt = st.chat_input('请输入问题或控制指令...')
