在 DeepEval 中进行最简单的 Q/A（问答）验证，通常使用的是 AnswerRelevancyMetric（回答相关性指标），它会评估模型生成的回答（actual_output）是否切合用户的提问（input）。

由于你是调用内部部署的模型，在 DeepEval 中使用自定义 LLM 主要有以下两种方案。请根据你内部接口的类型进行选择：

方案一（最省心，推荐）： 如果你的内部 API 兼容 OpenAI 接口格式（如使用 vLLM、Ollama、FastChat、TGI、OneAPI 等框架部署）。
方案二（最灵活）： 如果你的内部 API 是完全自定义的 HTTP 格式。
方案一：通过内置 GPTModel 快速接入（兼容 OpenAI 格式）
如果你的内部模型接口提供了类似 /v1/chat/completions 的 OpenAI 兼容端点，你不需要手写复杂的请求类，直接使用 DeepEval 内置的 GPTModel 并重定向 base_url 即可。

这是最简单、代码量最少的实现：

python
复制代码
from deepeval import evaluate
from deepeval.models import GPTModel
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# 1. 配置你的内部部署模型（使用内置的 GPTModel 进行桥接）
custom_model = GPTModel(
    model="your-internal-model-name",          # 你的内部模型名称
    api_key="any-string-or-your-key",          # 你的 API Key（如果没有可填任意字符串占位）
    base_url="http://192.168.1.100:8000/v1",   # 替换为你内部部署的 API 地址
    temperature=0.0                            # 评测时建议将温度设为 0 以保证结果一致性
)

# 2. 定义评估指标 (把自定义模型 custom_model 传入作为裁判)
relevancy_metric = AnswerRelevancyMetric(
    threshold=0.5,                             # 合格阈值
    model=custom_model,                        # 指定使用你的内部模型作为裁判
    include_reason=True                        # 是否输出判定得分的原因
)

# 3. 创建问答测试用例 (模拟一次 QA 记录)
test_case = LLMTestCase(
    input="请问你们支持 30 天无理由退货吗？",
    actual_output="支持的，所有商品在签收之日起 30 天内，只要不影响二次销售都可以办理退货。"
)

# 4. 执行评估并输出结果
evaluate(
    test_cases=[test_case],
    metrics=[relevancy_metric]
)

# 打印评测结果
print(f"评估得分: {relevancy_metric.score}")
print(f"评估理由: {relevancy_metric.reason}")
方案二：通过继承 DeepEvalBaseLLM 手动封装（非 OpenAI 格式）
如果你的内部 API 是完全自定义的（例如只接受特定的 JSON 格式请求体），你需要继承 DeepEvalBaseLLM 类并自己实现 generate 方法。

⚠️ 注意（避坑指南）：
DeepEval 的很多指标需要 LLM 裁判返回特定格式的 JSON。在自定义 generate 函数时，必须接收 schema: BaseModel = None 参数，并且当 DeepEval 传入 schema 时，你的模型生成完 JSON 后必须将其反序列化为该 schema 对应的 Pydantic 实例返回，否则会报错 AttributeError: 'str' object has no attribute 'statements'。

以下是完整的自定义实现代码：

python
复制代码
import os
import json
import requests
from pydantic import BaseModel
from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# 1. 继承 DeepEvalBaseLLM，封装你的内部 API
class MyCustomLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str, api_url: str, api_key: str = None):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key

    def load_model(self):
        # 简单返回自己或你的 API client 实例即可
        return self

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, schema: BaseModel = None) -> BaseModel | str:
        """
        同步生成方法。
        当 schema 不为 None 时，模型需要返回符合 schema (Pydantic 模型) 格式的数据结构。
        """
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # 2. 构造你内部 API 专属的请求体
        payload = {
            "model_name": self.model_name,
            "prompt_text": prompt,
            "temperature": 0.0
        }
        
        # 3. 发送请求
        response = requests.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()
        
        # 假设你的 API 返回格式为：{"result": "模型生成的文本"}
        result_json = response.json()
        raw_output = result_json.get("result", "")
        
        # 4. 关键：如果是 DeepEval 要求的结构化输出 (JSON)
        if schema:
            try:
                # 剔除可能存在的 markdown 代码块 (如 ```json ... ```)
                cleaned_output = raw_output.strip()
                if cleaned_output.startswith("```json"):
                    cleaned_output = cleaned_output[7:]
                if cleaned_output.endswith("```"):
                    cleaned_output = cleaned_output[:-3]
                cleaned_output = cleaned_output.strip()
                
                # 反序列化
                json_data = json.loads(cleaned_output)
                # 必须返回该 Pydantic BaseModel 的实例对象
                return schema(**json_data)
            except Exception as e:
                print(f"解析 JSON Schema 失败。模型原始输出：{raw_output}")
                raise e

        return raw_output

    async def a_generate(self, prompt: str, schema: BaseModel = None) -> BaseModel | str:
        """
        异步生成方法。为简单起见，这里直接复用同步生成方法。
        """
        return self.generate(prompt, schema)


# ================== 使用你自定义的类进行评测 ==================

# 1. 实例化你的内部自定义模型
custom_model = MyCustomLLM(
    model_name="my-internal-llama3",
    api_url="http://your-company-api-server/generate", # 替换为你公司的内部 API 地址
    api_key="your-custom-token"
)

# 2. 定义评估指标 (将你的模型传入作为裁判)
relevancy_metric = AnswerRelevancyMetric(
    threshold=0.5,
    model=custom_model,
    include_reason=True
)

# 3. 创建问答测试用例
test_case = LLMTestCase(
    input="请问你们支持 30 天无理由退货吗？",
    actual_output="支持的，所有商品在签收之日起 30 天内，只要不影响二次销售都可以办理退货。"
)

# 4. 执行评估
evaluate(
    test_cases=[test_case],
    metrics=[relevancy_metric]
)

print(f"评估得分: {relevancy_metric.score}")
print(f"评估理由: {relevancy_metric.reason}")
💡 进阶小贴士
如果你需要比对标准答案（Ground Truth）： 若不仅想验证“回答是否和提问相关”，还想验证“回答是否正确”（即需要和预期的标准答案比对），你可以在 LLMTestCase 中加入 expected_output="标准答案文本"，并换用 AnswerCorrectnessMetric 或直接使用灵活度极高的 GEval：
python
复制代码
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

correctness_metric = GEval(
    name="Correctness",
    criteria="判定 actual_output 是否与 expected_output 含义一致，不要遗漏关键信息。",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=custom_model
)
本地离线运行： DeepEval 默认会在本地计算并将数据存盘，如果你不需要上传结果到 Confident AI 的在线云平台看板，不需要在终端里运行 deepeval login，上面的代码在本地直接运行 python filename.py 即可完全离线执行。





