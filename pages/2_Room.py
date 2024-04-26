import streamlit as st

from pages.util.util import load_config, save_config

room_cfg = {
    'idx': 0,
    'name': '书生浦语智慧农业大棚',
    'description': '智慧大棚助手是一款基于物联网技术的智能农业解决方案，旨在帮助农户实现大棚环境的自动控制和优化。它通过连接各种传感器和执行器，可以实时监测大棚内的温度、湿度、光照等环境参数，并根据预设的规则和算法自动调节通风、遮阳、加热等设备，以保持适宜的作物生长环境。此外，智慧大棚助手还提供了远程监控和数据管理功能，方便农户随时随地了解大棚状态和作物生长情况。',
    'room': {},
}

if 'room' not in st.session_state:
    load_config()

if 'room' not in st.session_state:
    st.session_state['room'] = room_cfg

st.set_page_config(page_title='农业大棚设置 - 农业助手', layout='wide')

st.title('农业大棚设置')


name = st.text_input('大棚名', value=st.session_state['room']['name'], max_chars=None, key=None, type='default', placeholder = '请输入大棚名')
description = st.text_input('大棚描述', value=st.session_state['room']['description'], max_chars=None, key=None, type='default', placeholder = '请输入大棚描述')

if 0 == len(st.session_state['room']['room']):
    st.warning('当前还没有房间，请先创建大棚房间')
else:
    for k, v in st.session_state['room']['room'].items():
        with st.expander(v['name']):
            if 0 == len(v['drivers']):
                st.warning('当前房间还没有设备，请先添加设备')
            else:
                for dk, dv in v['drivers'].items():
                    inp = st.toggle(dv['name'], key = f'toggle_{k}_{dk}', value = dv['status'])

                    if inp:
                        st.session_state['room']['room'][k]['drivers'][dk]['status'] = True
                    else:
                        st.session_state['room']['room'][k]['drivers'][dk]['status'] = False

            col1, col2, col3, col4, col5 = st.columns(5)
            btn_create_lamp = col1.button('添加灯', key = f'btn_lamp_{k}')
            btn_create_spray = col2.button('添加喷淋', key = f'btn_spray_{k}')
            btn_create_fan = col3.button('添加通风', key = f'btn_fan_{k}')

            if btn_create_lamp:
                if 'lamp' in st.session_state['room']['room'][k]['idx']:
                    st.session_state['room']['room'][k]['idx']['lamp'] += 1
                else:
                    st.session_state['room']['room'][k]['idx']['lamp'] = 1

                driver_id = 'lamp_' + str(st.session_state['room']['room'][k]['idx']['lamp'])
                st.session_state['room']['room'][k]['drivers'][driver_id] = {
                    'type': 'lamp',
                    'name': str(st.session_state['room']['room'][k]['idx']['lamp']) + '号灯',
                    'id': driver_id,
                    'status': False
                }
            if btn_create_spray:
                if 'spray' in st.session_state['room']['room'][k]['idx']:
                    st.session_state['room']['room'][k]['idx']['spray'] += 1
                else:
                    st.session_state['room']['room'][k]['idx']['spray'] = 1

                driver_id = 'spray_' + str(st.session_state['room']['room'][k]['idx']['spray'])
                st.session_state['room']['room'][k]['drivers'][driver_id] = {
                    'type': 'spray',
                    'name': str(st.session_state['room']['room'][k]['idx']['spray']) + '号喷淋',
                    'id': driver_id,
                    'status': False
                }
            if btn_create_fan:
                if 'fan' in st.session_state['room']['room'][k]['idx']:
                    st.session_state['room']['room'][k]['idx']['fan'] += 1
                else:
                    st.session_state['room']['room'][k]['idx']['fan'] = 1

                driver_id = 'fan_' + str(st.session_state['room']['room'][k]['idx']['fan'])
                st.session_state['room']['room'][k]['drivers'][driver_id] = {
                    'type': 'fan',
                    'name': str(st.session_state['room']['room'][k]['idx']['fan']) + '号通风',
                    'id': driver_id,
                    'status': False
                }

col1, col2, col3, col4, col5 = st.columns(5)
btn_save = col1.button('保存')
btn_create_room = col2.button('创建房间')

if btn_save:
    st.session_state['room']['name'] = name
    st.session_state['room']['description'] = description

    save_config()

if btn_create_room:
    st.session_state['room']['idx'] += 1
    cell_name = str(st.session_state['room']['idx']) + '号房间'
    st.session_state['room']['room'][st.session_state['room']['idx']] = {'name': cell_name, 'drivers': {}, 'idx': {}}
