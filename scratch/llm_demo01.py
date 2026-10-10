#用于测试使用SDK的LLM连接
import os
import dotenv
from openai import OpenAI
# 加载环境变量
dotenv.load_dotenv()

# 创建 OpenAI 客户端
my_client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_API_BASE_URL"),
)


response = my_client.chat.completions.create(
    model=os.getenv("LLM_MODEL_NAME"),
    messages=[
        {"role": "user", "content": "解释下Redis是什么"}
    ]
)

# 打印响应  
print(response.choices[0].message.content)
