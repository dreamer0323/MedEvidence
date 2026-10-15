#用于测试使用SDK的LLM连接
import os
import dotenv
from openai import OpenAI
import time
# 加载环境变量
dotenv.load_dotenv()


def create_client(api_key: str = None, base_url: str = None)->OpenAI:
    """
    创建OpenAI客户端
    """
    api_key = api_key or os.getenv("LLM_API_KEY")
    base_url = base_url or os.getenv("LLM_API_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client


def invoke_llm(client: OpenAI, prompt: str,timeout: float = 30):
    """
    调用LLM模型
    """
    start_time = time.time()
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME"),
        messages=[
            {"role": "user", "content": prompt}
        ],
        timeout=timeout
    )
    end_time = time.time()
    if end_time - start_time > timeout:
        raise TimeoutError("LLM调用超时")
    print(f"LLM调用耗时: {end_time - start_time:.2f}秒")

    return response.choices[0].message.content

class myLLM:
    """
    自定义LLM类
    """
    def __init__(self, api_key: str = None, base_url: str = None,timeout: float = 30):
        self.client = create_client(api_key, base_url)
        self.timeout = timeout

    def invoke(self, prompt: str):
        """
        调用LLM模型
        """
        response = invoke_llm(self.client, prompt,self.timeout)
        return response





def main():
    """
    主函数
    """
    client = create_client()
    response = invoke_llm(client, "介绍下黑马点评")
    print(response)


if __name__ == "__main__":
    main()