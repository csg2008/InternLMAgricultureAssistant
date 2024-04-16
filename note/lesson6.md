# Lagent & AgentLego 智能体应用搭建

## 一、智能体概述
大模型智能体是一种具备自主智能的实体，它具有任务规划、记忆及调用外部工具等能力，能够识别出应该执行的任务以及执行方式，从而实现自主决策。Agent不仅仅是执行任务的工具，而是具有类似于有主观能动性的人的“仿主体性”，能够模仿人类的方式实现从“计划”、“组织”、“执行”、“学习”等过程直至完成一个任务。当前的Agent主要被分为三种类型：单体Agent、多Agent协作（Multi-Agent）、以及与人交互的Agent。基于大模型的AI Agent并非一种“新技术”，而更象是面向大型语言模型的一种“新的管理方式”，它使得大模型有能力处理更复杂、更多样化的决策任务，从而为大型模型从“语言”成功迈向“真实世界”。

CoT（Causal Thinking）在AI Agent中发挥了关键作用，它通过序列化的思考模式，让智能体能更有效地理解和处理输入信息。CoT帮助增强智能体进行决策的可靠性，将任务分解为一系列小步骤，然后通过逻辑推理，逐一解决这些小步骤，最终将这些解决方案整合为完成整个任务的方案。这种逐步解决问题的方法，可以大大降低在面对复杂任务时，因信息过多或过于复杂，而导致的错误决策。同时，这种方法也提高了整个解决方案的可追踪性和可验证性。

## 二、智能体基础环境部署
### A、创建 A100 10% Cuda12.2-conda 开发机

### B、安装基础环境，参考命令
```bash
studio-conda -t agent -o pytorch-2.1.2
```

### C、安装智能体框架，参考命令
```bash
conda activate agent
mkdir -p /root/agent
cd /root/agent && git clone https://gitee.com/internlm/lagent.git
cd /root/agent/lagent && git checkout 581d9fb && pip install -e .
cd /root/agent && git clone https://gitee.com/internlm/agentlego.git
cd /root/agent/agentlego && git checkout 7769e0d && pip install -e .
```

### D、安装其他依赖
在这一步中，我们将会安装其他将要用到的依赖库，如 LMDeploy，可以执行如下命令：
```bash
conda activate agent
pip install lmdeploy==0.3.0
```

### E、准备 Tutorial
由于后续的 Demo 需要用到 tutorial 已经写好的脚本，因此我们需要将 tutorial 通过 git clone 的方式克隆到本地，以备后续使用：
```bash
cd /root/agent
git clone -b camp2 https://gitee.com/internlm/Tutorial.git
```

## 三、Lagent 轻量级智能体框架
Lagent 是一个轻量级开源智能体框架，旨在让用户可以高效地构建基于大语言模型的智能体。同时它也提供了一些典型工具以增强大语言模型的能力。

### A、LMDeploy 启动 internlm2-chat-7b API Server：
参考命令：
```bash
conda activate agent
lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b \
        --server-name 0.0.0.0 \
        --server-port 23333 \
        --model-name internlm2-chat-7b
```

### B、Lagent ArxivSearch Web Demo 试用
1、启动 Web Demo, 参考命令：
```bash
conda activate agent
cd /root/agent/lagent/examples
streamlit run internlm2_agent_web_demo.py --server.address 127.0.0.1 --server.port 7860
```

2、在浏览器打开 demo 页面，并输入 api server 服务器地址后开始试用
> 搜索内容 `请帮我搜索 internlm2`
![](./asset/52.png)

> 搜索内容 `请帮我搜索 Multi Agent`
![](./asset/53.png)

> 搜索内容 `请帮我搜索十万个为什么`
![](./asset/54.png)

> 搜索内容 `帮我搜索学前教育`
![](./asset/55.png)

3、使用体会
> - 有搜索结果：通过 ArxivSearch 可以方便的进行论文搜索，只是搜索结果输出排版不够直观，可以对排版进行完善与内容进行摘要的话会比较有实用价值，尤其是搜索到多个结果的时候。
> - 无搜索结果：从搜索返回的 JSON 看是没有搜索结果，可是JSON后面的内容却是有的，还是跟主题有点相关的重复内容，这个就比较迷惑人

### D、Lagent Web Demo 自定义工具试用
Lagent 智能体的功能是有限的，可以通过自定义工具为智能体添加需要的功能来扩充它的使用体验。更多关于 Lagent 自定义工具开发实现可以参考 [技术文档](https://lagent.readthedocs.io/zh-cn/latest/tutorials/action.html)

1、创建工具源码 `vi /root/agent/lagent/lagent/actions/weather.py` 并粘贴以下源码：
```python
import json
import os
import requests
from typing import Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

class WeatherQuery(BaseAction):
    """Weather plugin for querying weather information."""

    def __init__(self,
                 key: Optional[str] = None,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)
        key = os.environ.get('WEATHER_API_KEY', key)
        if key is None:
            raise ValueError(
                'Please set Weather API key either in the environment '
                'as WEATHER_API_KEY or pass it as `key`')
        self.key = key
        self.location_query_url = 'https://geoapi.qweather.com/v2/city/lookup'
        self.weather_query_url = 'https://devapi.qweather.com/v7/weather/now'

    @tool_api
    def run(self, query: str) -> ActionReturn:
        """一个天气查询API。可以根据城市名查询天气信息。

        Args:
            query (:class:`str`): The city name to query.
        """
        tool_return = ActionReturn(type=self.name)
        status_code, response = self._search(query)
        if status_code == -1:
            tool_return.errmsg = response
            tool_return.state = ActionStatusCode.HTTP_ERROR
        elif status_code == 200:
            parsed_res = self._parse_results(response)
            tool_return.result = [dict(type='text', content=str(parsed_res))]
            tool_return.state = ActionStatusCode.SUCCESS
        else:
            tool_return.errmsg = str(status_code)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return

    def _parse_results(self, results: dict) -> str:
        """Parse the weather results from QWeather API.

        Args:
            results (dict): The weather content from QWeather API
                in json format.

        Returns:
            str: The parsed weather results.
        """
        now = results['now']
        data = [
            f'数据观测时间: {now["obsTime"]}',
            f'温度: {now["temp"]}°C',
            f'体感温度: {now["feelsLike"]}°C',
            f'天气: {now["text"]}',
            f'风向: {now["windDir"]}，角度为 {now["wind360"]}°',
            f'风力等级: {now["windScale"]}，风速为 {now["windSpeed"]} km/h',
            f'相对湿度: {now["humidity"]}',
            f'当前小时累计降水量: {now["precip"]} mm',
            f'大气压强: {now["pressure"]} 百帕',
            f'能见度: {now["vis"]} km',
        ]
        return '\n'.join(data)

    def _search(self, query: str):
        # get city_code
        try:
            city_code_response = requests.get(
                self.location_query_url,
                params={'key': self.key, 'location': query}
            )
        except Exception as e:
            return -1, str(e)
        if city_code_response.status_code != 200:
            return city_code_response.status_code, city_code_response.json()
        city_code_response = city_code_response.json()
        if len(city_code_response['location']) == 0:
            return -1, '未查询到城市'
        city_code = city_code_response['location'][0]['id']
        # get weather
        try:
            weather_response = requests.get(
                self.weather_query_url,
                params={'key': self.key, 'location': city_code}
            )
        except Exception as e:
            return -1, str(e)
        return weather_response.status_code, weather_response.json()
```

2、重新启动 WEB DEMO
```bash
conda activate agent
cd /root/agent/Tutorial/agent
export WEATHER_API_KEY=替换为你的和风天气KEY
streamlit run internlm2_weather_web_demo.py --server.address 127.0.0.1 --server.port 7860
```

3、在浏览器打开 demo 页面，并输入 api server 服务器地址后开始试用
> 查询上海的天气 `上海的天气？`
![](./asset/56.png)
> 查询莫斯科的天气 `莫斯科天气？`
![](./asset/57.png)
> 非天气查询 `我现在在上海`
![](./asset/58.png)
> 通过上下文干扰查询
![](./asset/59.png)

4、使用体会

以对话的方式与智能体交互比较符合当前人们的习惯，也应该是实现 AI Agent 的重要途径。
> 遇到的问题：先直接查询指定城市天气可以很好的返回结果，再继续与大模型聊天气相关的话题，然后查询天气就可能会被干扰到，最后连开始时直接查询天气也不能正常工作。

## 四、AgentLego：组装智能体“乐高”
AgentLego 是一个提供了多种开源工具 API 的多模态工具包，旨在像是乐高积木一样，让用户可以快速简便地拓展自定义工具，从而组装出自己的智能体。

### A、辅助工具安装与试用
1、安装辅助工具，参考命令：
```bash
conda activate agent
pip install openmim==0.3.9
mim install mmdet==3.3.0
```

2、下载测试素材, 参考命令:
```bash
cd /root/agent
wget http://download.openmmlab.com/agentlego/road.jpg
```

3、新建脚本 `vi /root/agent/direct_use.py`, 粘贴以下代码：
```python
import re

import cv2
from agentlego.apis import load_tool

# load tool
tool = load_tool('ObjectDetection', device='cuda')

# apply tool
visualization = tool('/root/agent/road.jpg')
print(visualization)

# visualize
image = cv2.imread('/root/agent/road.jpg')

preds = visualization.split('\n')
pattern = r'(\w+) \((\d+), (\d+), (\d+), (\d+)\), score (\d+)'

for pred in preds:
    name, x1, y1, x2, y2, score = re.match(pattern, pred).groups()
    x1, y1, x2, y2, score = int(x1), int(y1), int(x2), int(y2), int(score)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(image, f'{name} {score}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

cv2.imwrite('/root/agent/road_detection_direct.jpg', image)
```

4、执行脚本测试

参考命令 `python /root/agent/direct_use.py`, 执行成功可以看到以下文字输出：
```
truck (345, 428, 528, 599), score 83
car (771, 510, 837, 565), score 81
car (604, 518, 677, 569), score 75
person (866, 503, 905, 595), score 74
person (287, 513, 320, 596), score 74
person (964, 501, 999, 604), score 72
person (1009, 503, 1047, 602), score 69
person (259, 510, 279, 575), score 65
car (1074, 524, 1275, 691), score 64
person (993, 508, 1016, 597), score 62
truck (689, 483, 764, 561), score 62
bicycle (873, 551, 903, 602), score 60
person (680, 523, 699, 567), score 55
bicycle (968, 551, 996, 609), score 53
bus (826, 482, 930, 560), score 52
bicycle (1011, 551, 1043, 617), score 51
```

图像目标检测结果输出：
![](./asset/60.jpg)

### B、作为智能体工具使用
1、切换工具使用的模型为 internlm2-chat-7b

参考文档将 `/root/agent/agentlego/webui/modules/agents/lagent_agent.py` 中的模型由 `internlm2-chat-20b` 修改为 `internlm2-chat-7b`

2、启动 AgentLego WebUI
保持前面 Lagent 使用时的 API Server 运行，再在新标签中启动新的工具，参考命令：
```bash
conda activate agent
cd /root/agent/agentlego/webui
python one_click.py
```

3、使用 AgentLego WebUI Demo

在浏览器中打开工具 WebUI 界面，再按[参考教程](https://github.com/InternLM/Tutorial/blob/camp2/agent/agentlego.md#24-%E4%BD%BF%E7%94%A8-agentlego-webui)来配置工具, 配置好后可以切换到 chat 标签开始聊天使用，如下：
![](./asset/61.png)

4、使用体会与遇到的问题
> - 使用体验：用不同的图片与提示词重复试用了几十次，如果只选择一个工具调用成功的概率比较高，多选几个工具就会出现莫名奇妙的结果，比如：Calculator 会出现随机数学计算式，OCR 会识别出乱码，ObjectDetection 只检测出部分目标。可是只选择一个工具又有比较大的局限性，尤其是在不熟悉工具的具体功能的时候就只能一个一个工具的尝试，还不一定能写出相关性高的提示词。要是能自动根据图片与提示词来选择工具就好了。
> - 遇到的问题：上下文丢失，同一张图片每次提问题都必须重新上传一次; api server 崩溃后聊天可以看到发送的内容但没有回复也没有提示; 工具会间歇性不工作表现为机器人图标闪几下就没了，看日志输出就是处理时间明显短的不正常, 只能刷新页面才恢复。

工具不工作时的日志输出：

![](./asset/62.png)

### C、AgentLego 自定义工具
如果觉得 AgentLego 的工具不能满足要求，也可以尝试开发自定义的工具，AgentLego 在这方面提供了较为[详细的文档](https://agentlego.readthedocs.io/zh-cn/latest/modules/tool.html)。下面的示例是如何集成 MagicMaker，它是汇聚了优秀 AI 算法成果的免费 AI 视觉素材生成与创作平台。

1、创建工具

新建工具脚本 `vi /root/agent/agentlego/agentlego/tools/magicmaker_image_generation.py`，粘贴以下源码：
```python
import json
import requests

import numpy as np

from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import require
from .base import BaseTool

class MagicMakerImageGeneration(BaseTool):

    default_desc = ('This tool can call the api of magicmaker to '
                    'generate an image according to the given keywords.')

    styles_option = [
        'dongman',  # 动漫
        'guofeng',  # 国风
        'xieshi',   # 写实
        'youhua',   # 油画
        'manghe',   # 盲盒
    ]
    aspect_ratio_options = [
        '16:9', '4:3', '3:2', '1:1',
        '2:3', '3:4', '9:16'
    ]

    @require('opencv-python')
    def __init__(self,
                 style='guofeng',
                 aspect_ratio='4:3'):
        super().__init__()
        if style in self.styles_option:
            self.style = style
        else:
            raise ValueError(f'The style must be one of {self.styles_option}')

        if aspect_ratio in self.aspect_ratio_options:
            self.aspect_ratio = aspect_ratio
        else:
            raise ValueError(f'The aspect ratio must be one of {aspect_ratio}')

    def apply(self,
              keywords: Annotated[str,
                                  Info('A series of Chinese keywords separated by comma.')]
        ) -> ImageIO:
        import cv2
        response = requests.post(
            url='https://magicmaker.openxlab.org.cn/gw/edit-anything/api/v1/bff/sd/generate',
            data=json.dumps({
                "official": True,
                "prompt": keywords,
                "style": self.style,
                "poseT": False,
                "aspectRatio": self.aspect_ratio
            }),
            headers={'content-type': 'application/json'}
        )
        image_url = response.json()['data']['imgUrl']
        image_response = requests.get(image_url)
        image = cv2.imdecode(np.frombuffer(image_response.content, np.uint8), cv2.IMREAD_COLOR)
        return ImageIO(image)
```

2、将新创建的工具注册到可用工具列表中

修改 `/root/agent/agentlego/agentlego/tools/__init__.py`, 先导入工具再加入 `__all__` 列表中。修改后的文件内容如下：
```python
from .base import BaseTool
from .calculator import Calculator
from .func import make_tool
from .image_canny import CannyTextToImage, ImageToCanny
from .image_depth import DepthTextToImage, ImageToDepth
from .image_editing import ImageExpansion, ImageStylization, ObjectRemove, ObjectReplace
from .image_pose import HumanBodyPose, HumanFaceLandmark, PoseToImage
from .image_scribble import ImageToScribble, ScribbleTextToImage
from .image_text import ImageDescription, TextToImage
from .imagebind import AudioImageToImage, AudioTextToImage, AudioToImage, ThermalToImage
from .object_detection import ObjectDetection, TextToBbox
from .ocr import OCR
from .scholar import *  # noqa: F401, F403
from .search import BingSearch, GoogleSearch
from .segmentation import SegmentAnything, SegmentObject, SemanticSegmentation
from .speech_text import SpeechToText, TextToSpeech
from .translation import Translation
from .vqa import VQA
from .magicmaker_image_generation import MagicMakerImageGeneration

__all__ = [
    'CannyTextToImage', 'ImageToCanny', 'DepthTextToImage', 'ImageToDepth',
    'ImageExpansion', 'ObjectRemove', 'ObjectReplace', 'HumanFaceLandmark',
    'HumanBodyPose', 'PoseToImage', 'ImageToScribble', 'ScribbleTextToImage',
    'ImageDescription', 'TextToImage', 'VQA', 'ObjectDetection', 'TextToBbox', 'OCR',
    'SegmentObject', 'SegmentAnything', 'SemanticSegmentation', 'ImageStylization',
    'AudioToImage', 'ThermalToImage', 'AudioImageToImage', 'AudioTextToImage',
    'SpeechToText', 'TextToSpeech', 'Translation', 'GoogleSearch', 'Calculator',
    'BaseTool', 'make_tool', 'BingSearch','MagicMakerImageGeneration'
]
```

3、体验自定义工具效果

保持前面 Lagent 使用时的 API Server 运行，再在新标签中启动新的工具箱，参考命令：
```bash
conda activate agent
cd /root/agent/agentlego/webui
python one_click.py
```

工具箱启动好后在浏览器进入  AgentLego WebUI DEMO 界面继续使用，如下：
![](./asset/63.png)

4、使用体会与遇到的问题
> - 使用体验：MagicMakerImageGeneration 工具在随机创作方面蛮好玩，在指令关键词提取方面不怎么准确。
> - 遇到的问题：指令关键词提取不准确导致绘画风格不好把控; 同一个会话画了十多次后会出现没响应现象，只能刷新页面才恢复。

## 五、使用体会总结
基于智能体，可以很好的扩展大语言模型的应用范围，通过自定义工具与环境交互可以实现更好的个性化服务。智能体的使用也会涉及到一系列的挑战和机遇，如：泛化能力、交互效率、个性化、多智能体社会、安全性和评价方法等。这些问题需要通过技术创新和研究来解决，以实现AI Agent在更广泛的应用场景中发挥其价值。

未解疑问：
> - 如何更好的实现多个工具协？在使用 AgentLego Web Demo 钩选多个工具有时会出现工具误触发回复完全无关问题
> - 如何提升工具调用泛化能力与错误响应能力？Lagent 天气工具在被上下文干扰后就不能工作、AgentLego 在多轮对话后会出现没响应现象，从日志看也没有任何相关提示输出。