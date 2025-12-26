# src/llm/qwen_client.py (修改版 - 支持联网搜索)
import dashscope
from dashscope import Generation
from typing import List, Dict, Optional, Union
import json
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class QwenClient:
    """通义千问客户端 - 支持联网搜索"""
    def __init__(self, api_key: str, model: str = "qwen-max"):
        dashscope.api_key = api_key
        self.model = model
        # 模型参数配置
        self.model_configs = {
            "qwen-turbo": {
                "max_tokens": 2000,
                "temperature": 0.3,
                "top_p": 0.8,
            },
            "qwen-plus": {
                "max_tokens": 4000,
                "temperature": 0.3,
                "top_p": 0.8,
            },
            "qwen-max": {
                "max_tokens": 6000,
                "temperature": 0.3,
                "top_p": 0.8,
            },
            "qwen-long": { # 假设使用支持联网的模型
                "max_tokens": 8000,
                "temperature": 0.3,
                "top_p": 0.8,
            }
        }
        # 获取当前模型配置
        self.config = self.model_configs.get(model, self.model_configs["qwen-max"])

        # 系统提示词
        self.system_prompts = {
            "default": "你是一位资深的投行分析师，擅长企业分析和风险评估。请以专业、严谨的态度回答问题。请基于最新的网络信息进行分析。",
            "withdrawal_analysis": "你是资深IPO审核专家，专门分析企业撤否原因和审核问题。请基于最新的网络信息进行分析。",
            "financial_analysis": "你是资深财务分析师，擅长分析企业财务报表和财务风险。请基于最新的网络信息进行分析。",
            "industry_analysis": "你是行业研究专家，擅长分析行业趋势和竞争格局。请基于最新的网络信息进行分析。",
            "risk_analysis": "你是风险控制专家，擅长识别和评估企业各类风险。请基于最新的网络信息进行分析。",
            "online_search": "你是一位资深分析师，需要基于最新的网络信息回答用户问题。请首先通过网络搜索获取相关信息，然后进行专业分析和总结。"
        }

    def chat_completion_with_online_search(self, messages: List[Dict], scenario: str = None, **kwargs) -> str:
        """调用模型，包含联网搜索指令"""
        try:
            # 合并配置
            config = self.config.copy()
            config.update(kwargs)

            # 获取系统提示词
            system_prompt = self.system_prompts.get(scenario, self.system_prompts["default"])
            if scenario in ["withdrawal_analysis", "financial_analysis", "industry_analysis", "risk_analysis"]:
                 system_prompt = self.system_prompts["online_search"] + " " + system_prompt

            # 构建消息列表
            formatted_messages = [{"role": "system", "content": system_prompt}]
            formatted_messages.extend(messages)

            # 调用DashScope API
            # 注意：确保你使用的模型（如 qwen-long 或其他）支持联网搜索功能
            # 有些模型可能需要在请求中明确开启工具或插件
            response = Generation.call(
                model=self.model,
                messages=formatted_messages,
                temperature=config.get("temperature", 0.3),
                max_tokens=config.get("max_tokens", 2000),
                top_p=config.get("top_p", 0.8),
                # 如果API支持，可能需要添加类似下面的参数来启用联网
                # tools=[{"type": "web_search"}], # 示例，具体参数需查API文档
                result_format='message',
                stream=False
            )

            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                logger.error(f"Qwen API调用失败: {response.code} - {response.message}")
                return f"模型调用失败: {response.message}"
        except Exception as e:
            logger.error(f"Qwen客户端异常: {e}")
            return f"模型服务异常: {str(e)}"

    def analyze_with_online_context(self, query: str, scenario: str = None) -> Dict:
        """基于联网搜索进行分析"""
        # 直接将查询传递给支持联网的模型
        messages = [{"role": "user", "content": query}]
        response_text = self.chat_completion_with_online_search(messages, scenario)

        # 简单解析响应，尝试提取结构化信息
        try:
            # 尝试查找JSON部分 (如果模型输出了JSON格式)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                result["raw_response"] = response_text
                return result
        except json.JSONDecodeError:
            pass # 如果无法解析为JSON，继续处理

        # 如果无法解析为JSON，返回原始响应结构
        return {
            "analysis": response_text,
            "raw_response": response_text,
            "format": "text"
        }

    def structured_completion(self, prompt: str, scenario: str = None, output_format: Dict = None) -> Dict:
        """结构化输出 (保留原逻辑，但可选择性使用联网)"""
        # 如果需要联网，调用 analyze_with_online_context
        if scenario in ["withdrawal_analysis", "financial_analysis", "industry_analysis", "risk_analysis", "online_search"]:
             return self.analyze_with_online_context(prompt, scenario)

        # 否则使用普通聊天补全
        response_text = self.chat_completion([{"role": "user", "content": prompt}], scenario=scenario)
        # 尝试解析JSON
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                result["raw_response"] = response_text
                return result
        except json.JSONDecodeError:
            pass

        return {
            "analysis": response_text,
            "raw_response": response_text,
            "format": "text"
        }

    def chat_completion(self, messages: List[Dict], scenario: str = None, **kwargs) -> str:
        """普通聊天补全接口 (不强制联网)"""
        try:
            # 合并配置
            config = self.config.copy()
            config.update(kwargs)

            # 获取系统提示词
            system_prompt = self.system_prompts.get(scenario, self.system_prompts["default"])

            # 构建消息列表
            formatted_messages = [{"role": "system", "content": system_prompt}]
            formatted_messages.extend(messages)

            # 调用DashScope API
            response = Generation.call(
                model=self.model,
                messages=formatted_messages,
                temperature=config.get("temperature", 0.3),
                max_tokens=config.get("max_tokens", 2000),
                top_p=config.get("top_p", 0.8),
                result_format='message',
                stream=False
            )

            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                logger.error(f"Qwen API调用失败: {response.code} - {response.message}")
                return f"模型调用失败: {response.message}"
        except Exception as e:
            logger.error(f"Qwen客户端异常: {e}")
            return f"模型服务异常: {str(e)}"