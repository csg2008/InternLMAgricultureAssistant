
import streamlit as st
from lagent.actions.base_action import BaseAction, tool_api
from lagent.schema import ActionReturn, ActionStatusCode

class DeviceAssistant(BaseAction):
    """智慧大棚助手"""

    @tool_api(explode_return=True)
    def introduce(self) -> ActionReturn:
        """
        介绍智慧大棚助手

        Returns:
            :class:`str`: 大棚及房间信息
        """

        msg = []
        room = []
        tool_return = ActionReturn(type=self.name)

        if 'room' in st.session_state:
            msg.append('大棚名称：' + st.session_state['room']['name'])

            for _, v in st.session_state['room']['room'].items():
                room.append(v['name'])

            if len(room) > 0:
                msg.append('大棚房间：' + ",".join(room))
            else:
                msg.append('大棚还未配置房间')

            tool_return.result = [dict(type='text', content="\n".join(msg))]
            tool_return.state = ActionStatusCode.SUCCESS
        else:
            tool_return.errmsg = '大棚还未配置'
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return

    @tool_api(explode_return=True)
    def query_room(self, query: str) -> ActionReturn:
        """
        查询房间信息

        Args:
            query (:class:`str`): 要查询的房间名称

        Returns:
            :class:`str`: 房间信息
        """

        msg = []
        room_query_ok = False
        driver_category = {'spray': '喷淋', 'lamp': '灯', 'fan': '通风', 'water': '水泵'}
        tool_return = ActionReturn(type=self.name)

        def format_driver_status(status: bool) -> str:
            if status:
                return '开启'
            else:
                return '关闭'

        if 'room' in st.session_state:
            msg.append('大棚名称：' + st.session_state['room']['name'])

            for _, v in st.session_state['room']['room'].items():
                if query == v['name']:
                    room_query_ok = True
                    if len(v['drivers']) > 0:
                        msg.append('大棚房间：' + v['name'] + ' 有以下设备：')

                        for _, dv in v['drivers'].items():
                            driver_info = f'编号：{dv["id"]} 名称：{dv["name"]} 类型：{driver_category[dv["type"]]} 状态：{format_driver_status(dv["status"])}'
                            msg.append(driver_info)
                    else:
                        msg.append('大棚房间：' + v['name'] + ' 暂无设备')

            if not room_query_ok:
                msg.append('大棚房间：' + query + ' 不存在')

            tool_return.result = [dict(type='text', content="\n".join(msg))]
            tool_return.state = ActionStatusCode.SUCCESS
        else:
            tool_return.errmsg = '大棚还未配置'
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return

    @tool_api(explode_return=True)
    def driver_open(self, room: str, driver: str, state: str) -> ActionReturn:
        """
        开启/关闭房间内设备

        Args:
            room (:class:`str`): 房间名称
            driver (:class:`str`): 设备名称
            state (:class:`bool`): 开启/关闭

        Returns:
            :class:`str`: 操作状态
        """

        operator_state = False
        tool_return = ActionReturn(type=self.name)

        if 'room' in st.session_state:
            for _, v in st.session_state['room']['room'].items():
                if v['name'] == room:
                    for _, dv in v['drivers'].items():
                        if dv['name'] == driver or dv['id'] == driver:
                            if '开启' == state:
                                dv['status'] = True
                            else:
                                dv['status'] = False

                            operator_state = True
                            break

                break

            if operator_state:
                tool_return.result = [dict(type='text', content='操作成功')]
            else:
                tool_return.result = [dict(type='text', content='操作失败')]

            tool_return.state = ActionStatusCode.SUCCESS
        else:
            tool_return.errmsg = '大棚还未配置'
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return