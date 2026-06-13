"""LLaVA Agent 模型客户端"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import DEFAULT_IMAGE_TOKEN

from app.config import API_NAME_TO_WORKER, AGENT_URL, CONTROLLER_URL
from app.utils.dicom import encode_image_to_base64
from app.utils.http_client import get_http_session


class LLaVAAgentClient:
    """LLaVA Agent 模型客户端"""

    def __init__(self, controller_url: str = CONTROLLER_URL):
        self.controller_url = controller_url
        self.agent_url = None
        self.session = get_http_session()
        self._find_agent()

    def _find_agent(self):
        try:
            #先向controller发送获取可用模型的会话连接（TCP）
            resp = self.session.post(f"{self.controller_url}/list_models", timeout=5)
            models = resp.json().get("models", [])

            for model in models:
                if "llava" in model.lower() or "agent" in model.lower():
                    #把带有上面条件的模型打包返回
                    resp = self.session.post(
                        f"{self.controller_url}/get_worker_address",
                        json={"model": model},
                        timeout=5
                    )
                    self.agent_url = resp.json().get("address")
                    print(f"找到Agent模型: {model} at {self.agent_url}")
                    return

            self.agent_url = AGENT_URL
            print(f"使用默认Agent地址: {self.agent_url}")

        except Exception as e:
            print(f"查找Agent失败: {e}")
            self.agent_url = AGENT_URL

    def chat(self, prompt: str, images: List[str] = None, is_chinese: bool = False) -> Tuple[str, Optional[Dict]]:
        """与 Agent 进行单轮对话"""
        if not self.agent_url:
            return "Agent不可用", None

        conv = conv_templates["v1"].copy()
        if is_chinese:
            conv.system = "用户和人工智能助手之间的对话。助手必须用中文回答所有问题，包括思考过程、决策和最终结论都必须使用中文。助手需要对用户的问题给出有帮助、详细且礼貌的回答。"

        #如果对话的分隔符风格（conv.sep_style）不是 TWO（双分隔符模式），就使用默认分隔符 conv.sep；否则，就使用第二个分隔符 conv.sep2
        #（单分隔符模式）：人类和 AI 说完话后，用同一种符号隔开
        #双分隔符模式）：人类说完用第一种符号隔开，AI 说完用第二种符号隔开。
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        num_images = len(images) if images else 0
        if num_images > 0:
            #把图片转换成文本里的占位符
            #把用户原本文字里可能自带的、写错位置的 <image> 标签统统擦掉（replace）
            prompt_clean = prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()
            #根据图片的数量，生成对应数量的占位符。比如 2 张图就是 <image>\n<image>\n
            image_tokens = (DEFAULT_IMAGE_TOKEN + '\n') * num_images
            #把生成的图片占位符强行拼在用户清理过的文字最前面。现在提示词变成了类似：<image>\n请告诉我这张图里有什么？ 的标准格式。
            prompt = image_tokens + prompt_clean

        #conv.roles[0]：通常代表用户（Human / User）
        conv.append_message(conv.roles[0], prompt)
        #conv.roles[1]：通常代表 AI（Assistant）。
        conv.append_message(conv.roles[1], None)

        formatted_prompt = conv.get_prompt()

        pload = {
            "model": "agent",
            "prompt": formatted_prompt,
            "temperature": 0.2,
            "max_new_tokens": 1024,
            "stop": stop_str,
        }

        if images:
            encoded_images = []
            for img in images:
                if os.path.exists(img):
                    #如果是硬盘里的文件，就调用底层的 encode_image_to_base64 把图片转换成 Base64 编码
                    encoded_images.append(encode_image_to_base64(img))
                else:
                    encoded_images.append(img)
            pload["images"] = encoded_images

        try:
            resp = self.session.post(
                f"{self.agent_url}/worker_generate_stream",
                json=pload,
                stream=True,
                timeout=60
            )

            full_response = ""
            for chunk in resp.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data.get("error_code", 0) == 0:
                        #这里之所以能直接赋值覆盖，是因为底层的 LLaVA 模型每次吐出的 text 都是包含前面所有内容的完整句子，而不是只吐出一个新字
                        full_response = data.get("text", "")

            action = self._parse_action(full_response)
            return full_response, action

        except Exception as e:
            print(f"Agent请求失败: {e}")
            return str(e), None

    def chat_with_history(self, messages: List[Dict], images: List[str] = None, is_chinese: bool = False) -> Tuple[str, Optional[Dict]]:
        """与 Agent 进行多轮对话"""
        if not self.agent_url:
            return "Agent不可用", None

        conv = conv_templates["v1"].copy()
        if is_chinese:
            conv.system = "用户和人工智能助手之间的对话。助手必须用中文回答所有问题，包括思考过程、决策和最终结论都必须使用中文。助手需要对用户的问题给出有帮助、详细且礼貌的回答。"
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        num_images = len(images) if images else 0

        for i, msg in enumerate(messages):
            ## 判断角色：如果记录里写的是 "human"，就赋予模板里的用户角色（conv.roles[0]），否则赋予 AI 角色（conv.roles[1]）
            role = conv.roles[0] if msg["role"] == "human" else conv.roles[1]
            content = msg["content"]

            #把图片占位符只放在对话的第一句话前面（放在这整个对话历史（messages 列表）中的最开头的那句话（通常也就是用户发送的第一句话）前面。）
            if i == 0 and msg["role"] == "human" and num_images > 0:
                content_clean = content.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                image_tokens = (DEFAULT_IMAGE_TOKEN + '\n') * num_images
                content = image_tokens + content_clean

            conv.append_message(role, content)

        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        pload = {
            "model": "agent",
            "prompt": formatted_prompt,
            "temperature": 0.2,
            "max_new_tokens": 1024,
            "stop": stop_str,
        }

        if images:
            encoded_images = []
            for img in images:
                if os.path.exists(img):
                    encoded_images.append(encode_image_to_base64(img))
                else:
                    encoded_images.append(img)
            pload["images"] = encoded_images

        try:
            resp = self.session.post(
                f"{self.agent_url}/worker_generate_stream",
                json=pload,
                stream=True,
                timeout=60
            )

            full_response = ""
            for chunk in resp.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data.get("error_code", 0) == 0:
                        full_response = data.get("text", "")


            #full_response 包含了之前所有的历史对话，需要从一堆历史文本中，精确抠出 AI "最新" 回答的那句话
            if "ASSISTANT:" in full_response:
                # rsplit 表示从右边（末尾）开始切割，1 表示只切一次。
                # 这就确保我们拿到的是全篇对话中 *最后一次* ASSISTANT 说的话。
                last_assistant = full_response.rsplit("ASSISTANT:", 1)[-1]
                if last_assistant.strip():
                    full_response = last_assistant.strip()

            action = self._parse_action(full_response)
            return full_response, action

        except Exception as e:
            print(f"Agent请求失败: {e}")
            return str(e), None

    def _parse_action(self, response: str) -> Optional[Dict]:
        """解析 Agent 响应中的 action"""
        try:
            #找到 "thoughts🤔" 和 "actions🚀" 中间的一段话（这是模型的思考过程），再找到 "actions🚀" 和 "value👉" 中间的一段话（这是模型决定的动作）。
            pattern = r'"thoughts🤔"(.*?)"actions🚀"(.*?)"value👉"'

            #它会从左到右扫描整个字符串，把所有符合规则的匹配项全部揪出来，并打包成一个 Python 列表（List）返回给你。  若有多对阔号则返回元组
            # pattern (你想要寻找的正则表达式)   response (你要在哪段文本里找）
            #在默认情况下，正则表达式里的英文句号 . 代表“匹配任意字符，但不包括换行符 \n”。   加上 re.DOTALL 后，. 就拥有了“穿透力”，它可以匹配包括换行符在内的全宇宙任何字符！
            matches = re.findall(pattern, response, re.DOTALL)

            """例如
            [ ("我觉得应该切图。", '{"API_name": "seg"}') ]
            
            matches[0]：取出列表里的第一个（也是唯一一个）匹配项，即那个元组。    说是唯一的  原因是：由大模型的提示词决定的，即使你可以提取多个   我也只取第一个

            [1]：取出元组里的第 2 个元素，也就是代表 actions🚀 内容的那个 JSON 字符串！
            """

            if matches:
                actions_str = matches[0][1].strip()
                try:
                    #转换成 Python 字典
                    actions = json.loads(actions_str)
                    if actions and len(actions) > 0:
                        return actions[0]
                    else:
                        print(f"  [_parse_action] Agent返回空actions，进入Agent VQA模式")
                        return {"API_name": None, "API_params": {}, "no_api": True}
                except:
                    try:
                        #标准 JSON 规定字符串必须用双引号 "，但大模型经常抽风，吐出一个带有单引号 ' 的假 JSON
                        #replace("'", '"')，把单引号全换成双引号再试一次
                        actions = json.loads(actions_str.replace("'", '"'))
                        if actions and len(actions) > 0:
                            return actions[0]
                        else:
                            print(f"  [_parse_action] Agent返回空actions，进入Agent VQA模式")
                            return {"API_name": None, "API_params": {}, "no_api": True}
                    except:
                        pass

            #如果 actions🚀 这种 Emoji 格式都没输出，
            for api_name in API_NAME_TO_WORKER.keys():
                #直接拿系统里所有注册好的 API 名字（从字典 API_NAME_TO_WORKER.keys() 里取，比如 "HeartSeg2CHWorker"），在整个模型的回答文本里进行字符串暴力搜索（in response）
                if api_name in response:
                    return {"API_name": api_name, "API_params": {}}

            return None

        except Exception:
            return None

    def _parse_value(self, response: str) -> str:
        """解析 Agent 响应中的 value 部分（取最后一个匹配）"""
        try:
            #"value👉"(.*?) 提取 "value👉" 后面的内容
            #(?=...) (Positive Lookahead / 正向先行断言)   一直往后匹配，直到你【看到】括号里的这些词为止，但不要把这些词吃进去！
            #只要看到 "thoughts🤔"（开始下一轮思考）、或者 "actions🚀"（开始下一个动作）、或者 "value👉"（重复输出）、或者 $（整个字符串彻底结束），匹配就立刻踩刹车
            pattern = r'"value👉"(.*?)(?="thoughts🤔"|"actions🚀"|"value👉"|$)'
            matches = re.findall(pattern, response, re.DOTALL)

            if matches:
                #因为大模型有可能会生成多次结果，
                return matches[-1].strip().strip('"').strip()
            return response
        except:
            return response
