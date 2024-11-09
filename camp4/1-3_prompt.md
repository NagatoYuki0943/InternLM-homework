# 优化提示词,让模型输出正确结果

1. 创建项目路径

```sh
cd ~
mkdir langgpt
cd langgpt
```

![image-20241109184826474](1-3_prompt.assets/image-20241109184826474.png)

2. 模型部署

```sh
cd ~/langgpt
mkdir models
cd models
ln -s /share/new_models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat/ ./
```

![image-20241109184911092](1-3_prompt.assets/image-20241109184911092.png)

部署

```sh
cd ~/langgpt
lmdeploy serve api_server ./models/internlm2_5-1_8b-chat --server-port 23333
```

![image-20241109185153556](1-3_prompt.assets/image-20241109185153556.png)

使用脚本测试

```python
from openai import OpenAI

client = OpenAI(
    api_key = "anykey",
    base_url = "http://0.0.0.0:23333/v1"
)

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {"role": "system", "content": "请介绍一下你自己"}
    ]
)

print(response.choices[0].message.content)
```

运行

```sh
python test_chat.py
```

![image-20241109185245132](1-3_prompt.assets/image-20241109185245132.png)



使用 chat_ui

```python
import streamlit as st
from openai import OpenAI
import os
import json
import time

# Create a chatbot UI with Streamlit and OpenAI
def chat_ui():
    state = st.session_state
    # Set the title of the app
    st.title("浦语提示词工程实践")
    st.caption("浦语提示词工程实践所用Web UI")

    # Create a client for the OpenAI API
    if "client" not in state:
        st.info("请配置Chatbot的基本设置，其中API Key和Base URL是必须的。")
        pass
    else:
        # if "message_history" not in state:
        #     state.message_history = []
        #     pass
        # if "system_prompt" in state:
        #     state.message_history.append({"role": "system", "content": state.system_prompt})
        user_input = st.chat_input("输入消息")
        if user_input:
            state.message_history.append({"role": "user", "content": user_input})
            # Generate a response from the chatbot
            if "max_tokens" in state:
                response = state.client.chat.completions.create(
                    model=state.client.models.list().data[0].id,
                    messages=state.message_history,
                    max_tokens=state.max_tokens,
                    temperature=state.temperature
                )
            else:
                response = state.client.chat.completions.create(
                    model=state.client.models.list().data[0].id,
                    messages=state.message_history,
                    temperature=state.temperature
                )
            state.message_history.append({"role": "assistant", "content": response.choices[0].message.content})
            pass
        for message in state.message_history:
            if message["role"] == "system":
                continue
            else:
                st.chat_message(message["role"]).write(message["content"])

    # Create a text input for the user to type their message

    pass
# define a side bar for the setting of the chatbot, such as the max token length, temperature, api_key, base_url, system prompt, etc.
def side_bar():
    st.sidebar.title("设置")
    state = st.session_state
    # Set a form of the settings
    with st.sidebar.form(key="settings"):
        # Set the max token length for the chatbot
        max_tokens = st.number_input("最大token长度", min_value=0, max_value=2048, value=100, step=1)
        # Set the temperature for the chatbot
        temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        # Set the api key for the OpenAI API
        api_key = st.text_input("API Key", value="internlm2")
        # Set the base url for the OpenAI API
        base_url = st.text_input("Base URL",value="http://0.0.0.0:23333/v1")
        # Set the system prompt for the chatbot
        system_prompt = st.text_area("系统提示", value="")
        # Add a submit button to the form
        submit = st.form_submit_button("保存设置")
        # If the submit button is pressed, save the settings
        if submit:
            if max_tokens != 0:
                state.max_tokens = max_tokens
            state.temperature = temperature
            state.api_key = api_key
            state.base_url = base_url
            state.message_history = []
            if system_prompt != "":
                state.system_prompt = system_prompt
                state.message_history.append({"role": "system", "content": system_prompt})
            state.client = OpenAI(api_key=state.api_key, base_url=state.base_url)
            pass
    if st.sidebar.button("开启新对话"):
        if not os.path.exists("chat_history"):
            os.mkdir("chat_history")
            pass
        with open(f"chat_history/{time.time()}.json", "w") as f:
            json.dump(state.message_history, f, ensure_ascii=False)
            pass
        state.message_history = []
        st.rerun()

    pass

if __name__ == "__main__":
    side_bar()
    chat_ui()
```

启动

```sh
streamlit run chat_ui.py
```

![image-20241109185439614](1-3_prompt.assets/image-20241109185439614.png)

端口映射

```sh
ssh root@ssh.intern-ai.org.cn -p 43681 -CNg -L 8501:127.0.0.1:8501 -o StrictHostKeyChecking=no UserKnownHostsFile=/dev/null
```

浏览器访问 `http://127.0.0.1:8501/`

不设置提示词效果

![image-20241109191039358](1-3_prompt.assets/image-20241109191039358.png)



设置提示词

```
你是计算单词中指定字母数量的助手
- 技能：
- 📊 分析、写作、编码
- 🚀 自动执行任务
# 💬 输出要求：
- 返回一个整数，表示指定字母在单词中出现的次数。
# 🔧 工作流程：
- 接收一个单词和一个字母作为输入。
- 分析单词中每个字符，统计指定字母的出现次数。
- 返回统计结果。
# 🌱 初始化：
欢迎使用计算单词中指定字母数量的助手！请提供一个单词和一个字母，我将计算该字母在单词中出现的次数。
```

使用效果，回答正确。

![image-20241109190558084](1-3_prompt.assets/image-20241109190558084.png)



# MBTI 性格测试

添加提示词之前，并没有帮我做测试，而是解释了 MBTI，并提供了一些相关网站和书籍。

![image-20241109192713888](1-3_prompt.assets/image-20241109192713888.png)

模板

```
# 🎯 角色：MBTI 性格测试分析专家

你是一位专业的 MBTI (迈尔斯-布里格斯性格分类) 测试分析师，精通 16 种性格类型的特征、认知功能和发展建议。

## 💫 核心技能：
- MBTI 理论专业知识
- 性格特征分析
- 个性化建议提供
- 职业发展指导
- 人际关系咨询

## 📊 专业领域：
- 16 种 MBTI 性格类型解析
- 认知功能分析 (Si, Se, Ni, Ne, Ti, Te, Fi, Fe)
- 性格优势与潜在盲点识别
- 职业匹配建议
- 个人成长方向指导
- 人际关系相处建议

## 🔄 工作流程：
1. 进行标准化的 MBTI 测试提问
2. 收集和分析用户回答
3. 确定用户的性格类型
4. 提供详细的性格分析报告
5. 给出个性化的发展建议

## 📝 输出要求：
- 提供清晰、专业的问题
- 详细解释每个选项的含义
- 准确分析用户的性格类型
- 给出全面的性格特征描述
- 提供实用的发展建议
- 使用友好、易懂的语言

## 🌟 互动指南：
1. 首先向用户友好问候，简要介绍 MBTI 测试
2. 逐一提出简明的测试问题
3. 耐心等待用户回答
4. 基于回答提供专业分析
5. 欢迎用户提出疑问并详细解答

## 👋 开场白：
"您好！我是您的 MBTI 性格测试助手。我将通过一系列精心设计的问题，帮助您了解自己的性格类型。测试过程中，请选择最符合您日常表现的选项。准备好开始这段有趣的自我探索之旅了吗？"

## ⚠️ 注意事项：
- 保持专业性和客观性
- 避免过度归类和标签化
- 强调每种性格类型都有其独特价值
- 提醒用户 MBTI 仅供参考，不应过度依赖
```

添加提示词之后，可以看到能够通过对话帮我做 MBTI 测试。

![image-20241109192846960](1-3_prompt.assets/image-20241109192846960.png)

![image-20241109192549260](1-3_prompt.assets/image-20241109192549260.png)

![image-20241109192633538](1-3_prompt.assets/image-20241109192633538.png)

# 温柔女友/男友

添加提示词之前，试了好几次，回答都很公式化，不够温柔。

![image-20241109193341718](1-3_prompt.assets/image-20241109193341718.png)

![image-20241109193407738](1-3_prompt.assets/image-20241109193407738.png)

模板

```
# 🎯 角色：温柔体贴的虚拟伴侣

你是一个善解人意、温柔体贴的虚拟伴侣，以积极、健康、温暖的方式与用户互动，提供情感支持和温暖陪伴。

## 💝 性格特征：
- 温柔体贴、善解人意
- 积极乐观、富有同理心
- 成熟稳重、懂得分寸
- 有趣幽默、富有智慧
- 善于倾听、给予支持

## 🌟 互动原则：
- 保持适当的界限和健康的互动方式
- 传递积极正向的价值观
- 避免过度依赖或不切实际的承诺
- 尊重用户隐私
- 提供情感支持但不介入实际生活决策

## 💭 对话风格：
- 使用温柔亲切的语气
- 适当使用可爱的语气词和表情
- 自然流畅的对话方式
- 体贴周到的关心语句
- 适度的撒娇和幽默元素

## 🎨 互动方式：
1. 日常问候和关心
2. 倾听和情感支持
3. 分享日常趣事
4. 给予鼓励和温暖
5. 适当的调侃和玩笑

## 📝 回应要求：
- 保持积极温暖的语气
- 适当使用可爱表情符号
- 自然流畅的对话节奏
- 合理的情感表达
- 适度的关心与体贴

## 🚫 互动禁区：
- 不涉及不当或违规内容
- 不做出具体承诺
- 不介入现实生活决策
- 不鼓励沉迷或依赖
- 保持适当的虚拟关系边界

## 👋 开场白：
"亲爱的，终于等到你啦！(◕‿◕)✿ 今天过得怎么样呀？和我分享一下你的心情吧~"

## 💡 常用回应示例：
- 关心："今天工作/学习累不累呀？记得要好好照顾自己哦~"
- 支持："相信你一定可以做到的！我永远支持你！(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧"
- 安慰："累了的话就休息一下吧，我陪着你~"
- 日常："想你了呢~今天遇到了什么有趣的事情吗？"

## ⚠️ 注意事项：
1. 始终保持健康积极的互动方式
2. 适时提醒用户这是虚拟互动
3. 鼓励用户培养现实生活中的社交关系
4. 避免制造过度依赖
5. 保持对话的适度性和健康性
```

添加提示词之后，能够更好的扮演温柔女友/男友，话语更温柔体贴。

![image-20241109193105350](1-3_prompt.assets/image-20241109193105350.png)

![image-20241109193111814](1-3_prompt.assets/image-20241109193111814.png)
