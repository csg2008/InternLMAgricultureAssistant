import streamlit as st

room_cfg = {
    'idx': 0,
    'name': '农业大棚',
    'room': {},
}

if 'room' not in st.session_state:
    st.session_state['room'] = room_cfg

st.set_page_config(page_title='农业大棚设置 - 农业助手', layout='wide')

st.title('农业大棚设置')


name = st.text_input('大棚名', value=st.session_state['room']['name'], max_chars=None, key=None, type='default', placeholder = '请输入大棚名')

if 0 == len(st.session_state['room']['room']):
    st.warning('当前还没有房间，请先创建大棚房间')
else:
    for k, v in st.session_state['room']['room'].items():
        with st.expander(v['name']):
            if 0 == len(v['drivers']):
                st.warning('当前房间还没有设备，请先添加设备')
            else:
                for dk, dv in v['drivers'].items():
                    inp = st.toggle(dv['name'], key = f'toggle_{k}_{dk}')

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

if btn_create_room:
    st.session_state['room']['idx'] += 1
    cell_name = str(st.session_state['room']['idx']) + '号房间'
    st.session_state['room']['room'][st.session_state['room']['idx']] = {'name': cell_name, 'drivers': {}, 'idx': {}}
