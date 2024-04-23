import copy
import json
from typing import List

import streamlit as st

from lagent.actions import ArxivSearch, ActionExecutor, IPythonInterpreter
from lagent.agents.internlm2_agent import Internlm2Agent, Internlm2Protocol
from lagent.llms import HFTransformer
from lagent.llms.meta_template import INTERNLM2_META as META
from lagent.schema import AgentStatusCode
from lagent.actions.base_action import BaseAction

from action.weather import WeatherQuery

class StreamlitUI:
    """Streamlit UI class."""

    def clear_state(self):
        """Clear the existing session state."""
        st.session_state['model']['user'] = []
        st.session_state['model']['assistant'] = []

        if 'chatbot' in st.session_state:
            st.session_state['chatbot']._session_history = []

    def get_actions(self) -> List[BaseAction]:
        """Get the plugin actions."""

        return [
            ArxivSearch(),
            WeatherQuery(),
    ]

    def initialize_chatbot(self):
        """Initialize the chatbot with the given model and plugin actions."""
        if 'chatbot' not in st.session_state:
            model = HFTransformer(
                path=st.session_state['model']['model_path'],
                meta_template=META,
                max_new_tokens=1024,
                top_p=0.8,
                top_k=None,
                temperature=0.1,
                repetition_penalty=1.0,
                stop_words=['<|im_end|>']
            )

            st.session_state['chatbot'] = Internlm2Agent(
                llm=model,
                plugin_executor=ActionExecutor(actions=self.get_actions()),
                interpreter_executor = ActionExecutor(actions=[IPythonInterpreter()]),
                protocol=Internlm2Protocol(
                    meta_prompt=st.session_state['model']['prompt_meta'],
                    plugin_prompt=st.session_state['model']['prompt_plugin'],
                    interpreter_prompt=st.session_state['model']['prompt_da'],
                    tool=dict(
                        begin='{start_token}{name}\n',
                        start_token='<|action_start|>',
                        name_map=dict(
                            plugin='<|plugin|>', interpreter='<|interpreter|>'),
                        belong='assistant',
                        end='<|action_end|>\n',
                    ), ),
                max_turn=7
            )

    def render_user(self, prompt: str):
        """Render the user prompt in the Streamlit UI."""
        with st.chat_message('user'):
            st.markdown(prompt)

    def render_assistant(self, agent_return):
        """Render the assistant response in the Streamlit UI."""
        with st.chat_message('assistant'):
            for action in agent_return.actions:
                if (action) and (action.type != 'FinishAction'):
                    self.render_action(action)
            st.markdown(agent_return.response)

    def render_plugin_args(self, action):
        """Render the plugin arguments in the Streamlit UI."""
        action_name = action.type
        args = action.args

        parameter_dict = dict(name=action_name, parameters=args)
        parameter_str = '```json\n' + json.dumps(parameter_dict, indent=4, ensure_ascii=False) + '\n```'
        st.markdown(parameter_str)

    def render_interpreter_args(self, action):
        """Render the interpreter arguments in the Streamlit UI."""
        st.info(action.type)
        st.markdown(action.args['text'])

    def render_action(self, action):
        """Render the action in the Streamlit UI."""
        st.markdown(action.thought)
        if action.type == 'IPythonInterpreter':
            self.render_interpreter_args(action)
        elif action.type == 'FinishAction':
            pass
        else:
            self.render_plugin_args(action)
        self.render_action_results(action)

    def render_action_results(self, action):
        """Render the results of action, including text, images, videos, and audios."""
        if (isinstance(action.result, dict)):
            if 'text' in action.result:
                st.markdown('```\n' + action.result['text'] + '\n```')
            if 'image' in action.result:
                # image_path = action.result['image']
                for image_path in action.result['image']:
                    image_data = open(image_path, 'rb').read()
                    st.image(image_data, caption='Generated Image')
            if 'video' in action.result:
                video_data = action.result['video']
                video_data = open(video_data, 'rb').read()
                st.video(video_data)
            if 'audio' in action.result:
                audio_data = action.result['audio']
                audio_data = open(audio_data, 'rb').read()
                st.audio(audio_data)
        elif isinstance(action.result, list):
            for item in action.result:
                if item['type'] == 'text':
                    st.markdown('```\n' + item['content'] + '\n```')
                elif item['type'] == 'image':
                    image_data = open(item['content'], 'rb').read()
                    st.image(image_data, caption='Generated Image')
                elif item['type'] == 'video':
                    video_data = open(item['content'], 'rb').read()
                    st.video(video_data)
                elif item['type'] == 'audio':
                    audio_data = open(item['content'], 'rb').read()
                    st.audio(audio_data)
        if action.errmsg:
            st.error(action.errmsg)

def init_ui():
    """Initialize Streamlit UI and setup sidebar"""

    if 'ui' not in st.session_state:
        st.session_state['ui'] = StreamlitUI()

    st.session_state['ui'].initialize_chatbot()

    for prompt, agent_return in zip(st.session_state['model']['user'], st.session_state['model']['assistant']):
        st.session_state['ui'].render_user(prompt)
        st.session_state['ui'].render_assistant(agent_return)

    if user_input := st.chat_input('请输入问题或控制指令...'):
        with st.container():
            st.session_state['ui'].render_user(user_input)
        st.session_state['model']['user'].append(user_input)

        if isinstance(user_input, str):
            user_input = [dict(role='user', content=user_input)]
        st.session_state['model']['last_status'] = AgentStatusCode.SESSION_READY
        for agent_return in st.session_state['chatbot'].stream_chat(
                st.session_state['model']['session_history'] + user_input):
            if agent_return.state == AgentStatusCode.PLUGIN_RETURN:
                with st.container():
                    st.session_state['ui'].render_plugin_args(agent_return.actions[-1])
                    st.session_state['ui'].render_action_results(agent_return.actions[-1])
            elif agent_return.state == AgentStatusCode.CODE_RETURN:
                with st.container():
                    st.session_state['ui'].render_action_results(agent_return.actions[-1])
            elif (agent_return.state == AgentStatusCode.STREAM_ING
                  or agent_return.state == AgentStatusCode.CODING):
                # st.markdown(agent_return.response)
                # 清除占位符的当前内容，并显示新内容
                with st.container():
                    if agent_return.state != st.session_state['model']['last_status']:
                        st.session_state['model']['temp'] = ''
                        placeholder = st.empty()
                        st.session_state['model']['placeholder'] = placeholder
                    if isinstance(agent_return.response, dict):
                        action = f"\n\n {agent_return.response['name']}: \n\n"
                        action_input = agent_return.response['parameters']
                        if agent_return.response['name'] == 'IPythonInterpreter':
                            action_input = action_input['command']
                        response = action + action_input
                    else:
                        response = agent_return.response
                    st.session_state['model']['temp'] = response
                    st.session_state['model']['placeholder'].markdown(
                        st.session_state['model']['temp'])
            elif agent_return.state == AgentStatusCode.END:
                st.session_state['model']['session_history'] += (
                    user_input + agent_return.inner_steps)
                agent_return = copy.deepcopy(agent_return)
                agent_return.response = st.session_state['model']['temp']
                st.session_state['model']['assistant'].append(copy.deepcopy(agent_return))
            st.session_state['model']['last_status'] = agent_return.state

st.set_page_config(page_title='智慧大棚中心 - 农业助手', layout='wide')
st.title('🤠 智慧大棚中心')

if 'model' not in st.session_state:
    with st.container():
        st.warning('请先配置书生·浦语模型参数')
elif 'room' not in st.session_state or 'room' not in st.session_state['room'] or 0 == len(st.session_state['room']['room']):
    with st.container():
        st.warning('请先配置大棚参数')
else:
    init_ui()
