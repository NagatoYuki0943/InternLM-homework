# MindSearch CPU-only 版部署

1. 创建开发机 & 环境配置

```sh
mkdir -p /root/mindsearch
cd /root/mindsearch
git clone https://github.com/InternLM/MindSearch.git
cd MindSearch && git checkout b832275 && cd ..
```

![image-20240822195546826](2-6_MindSearch.assets/image-20240822195546826.png)

2. 创建一个 conda 环境来安装相关依赖(这里使用之前的环境)

```sh
# 激活环境
conda activate langgpt
# 安装依赖
pip install -r /root/mindsearch/MindSearch/requirements.txt
```

![image-20240822201955214](2-6_MindSearch.assets/image-20240822201955214.png)

3. 获取硅基流动 API Key

4. 启动 MindSearch

```sh
export SILICON_API_KEY=mykey
conda activate langgpt
cd /root/mindsearch/MindSearch
python -m mindsearch.app --lang cn --model_format internlm_silicon --search_engine DuckDuckGoSearch
```

![image-20240822202101908](2-6_MindSearch.assets/image-20240822202101908.png)

5. 启动前端

```sh
conda activate langgpt
cd /root/mindsearch/MindSearch
python frontend/mindsearch_gradio.py
```

![image-20240822202216623](2-6_MindSearch.assets/image-20240822202216623.png)

6. 映射端口

我们把 8002 端口和 7882 端口都映射到本地。

```
ssh -CNg -L 8002:127.0.0.1:8002 -L 7882:127.0.0.1:7882 root@ssh.intern-ai.org.cn -p 43681
```

![image-20240822202223188](2-6_MindSearch.assets/image-20240822202223188.png)

7. 访问 127.0.0.1:7882

问题: python 的 getattr 和 setattr 的第一个参数为 object，但是可以直接传入一个 class，修改class中的变量，class本身也算一个object吗

正确给出了答案

![image-20240822202715861](2-6_MindSearch.assets/image-20240822202715861.png)



# 部署到 HuggingFace Space

1. 创建 space

![image-20240822202955239](2-6_MindSearch.assets/image-20240822202955239.png)

2. 添加 secret key

![image-20240822203403879](2-6_MindSearch.assets/image-20240822203403879.png)

3. 新建一个目录，准备提交到 HuggingFace Space 的全部文件

```sh
# 创建新目录
mkdir -p /root/mindsearch/mindsearch_deploy
# 准备复制文件
cp -r /root/mindsearch/MindSearch/mindsearch /root/mindsearch/mindsearch_deploy
cp /root/mindsearch/MindSearch/requirements.txt /root/mindsearch/mindsearch_deploy
# 创建 app.py 作为程序入口
touch /root/mindsearch/mindsearch_deploy/app.py
```

`requirements.txt` 中添加

```sh
griffe==0.48
```

app.py 代码如下

```python
import json
import os

import gradio as gr
import requests
from lagent.schema import AgentStatusCode

os.system("python -m mindsearch.app --lang cn --model_format internlm_silicon &")

PLANNER_HISTORY = []
SEARCHER_HISTORY = []


def rst_mem(history_planner: list, history_searcher: list):
    '''
    Reset the chatbot memory.
    '''
    history_planner = []
    history_searcher = []
    if PLANNER_HISTORY:
        PLANNER_HISTORY.clear()
    return history_planner, history_searcher


def format_response(gr_history, agent_return):
    if agent_return['state'] in [
            AgentStatusCode.STREAM_ING, AgentStatusCode.ANSWER_ING
    ]:
        gr_history[-1][1] = agent_return['response']
    elif agent_return['state'] == AgentStatusCode.PLUGIN_START:
        thought = gr_history[-1][1].split('```')[0]
        if agent_return['response'].startswith('```'):
            gr_history[-1][1] = thought + '\n' + agent_return['response']
    elif agent_return['state'] == AgentStatusCode.PLUGIN_END:
        thought = gr_history[-1][1].split('```')[0]
        if isinstance(agent_return['response'], dict):
            gr_history[-1][
                1] = thought + '\n' + f'```json\n{json.dumps(agent_return["response"], ensure_ascii=False, indent=4)}\n```'  # noqa: E501
    elif agent_return['state'] == AgentStatusCode.PLUGIN_RETURN:
        assert agent_return['inner_steps'][-1]['role'] == 'environment'
        item = agent_return['inner_steps'][-1]
        gr_history.append([
            None,
            f"```json\n{json.dumps(item['content'], ensure_ascii=False, indent=4)}\n```"
        ])
        gr_history.append([None, ''])
    return


def predict(history_planner, history_searcher):

    def streaming(raw_response):
        for chunk in raw_response.iter_lines(chunk_size=8192,
                                             decode_unicode=False,
                                             delimiter=b'\n'):
            if chunk:
                decoded = chunk.decode('utf-8')
                if decoded == '\r':
                    continue
                if decoded[:6] == 'data: ':
                    decoded = decoded[6:]
                elif decoded.startswith(': ping - '):
                    continue
                response = json.loads(decoded)
                yield (response['response'], response['current_node'])

    global PLANNER_HISTORY
    PLANNER_HISTORY.append(dict(role='user', content=history_planner[-1][0]))
    new_search_turn = True

    url = 'http://localhost:8002/solve'
    headers = {'Content-Type': 'application/json'}
    data = {'inputs': PLANNER_HISTORY}
    raw_response = requests.post(url,
                                 headers=headers,
                                 data=json.dumps(data),
                                 timeout=20,
                                 stream=True)

    for resp in streaming(raw_response):
        agent_return, node_name = resp
        if node_name:
            if node_name in ['root', 'response']:
                continue
            agent_return = agent_return['nodes'][node_name]['detail']
            if new_search_turn:
                history_searcher.append([agent_return['content'], ''])
                new_search_turn = False
            format_response(history_searcher, agent_return)
            if agent_return['state'] == AgentStatusCode.END:
                new_search_turn = True
            yield history_planner, history_searcher
        else:
            new_search_turn = True
            format_response(history_planner, agent_return)
            if agent_return['state'] == AgentStatusCode.END:
                PLANNER_HISTORY = agent_return['inner_steps']
            yield history_planner, history_searcher
    return history_planner, history_searcher


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">MindSearch Gradio Demo</h1>""")
    gr.HTML("""<p style="text-align: center; font-family: Arial, sans-serif;">MindSearch is an open-source AI Search Engine Framework with Perplexity.ai Pro performance. You can deploy your own Perplexity.ai-style search engine using either closed-source LLMs (GPT, Claude) or open-source LLMs (InternLM2.5-7b-chat).</p>""")
    gr.HTML("""
    <div style="text-align: center; font-size: 16px;">
        <a href="https://github.com/InternLM/MindSearch" style="margin-right: 15px; text-decoration: none; color: #4A90E2;">🔗 GitHub</a>
        <a href="https://arxiv.org/abs/2407.20183" style="margin-right: 15px; text-decoration: none; color: #4A90E2;">📄 Arxiv</a>
        <a href="https://huggingface.co/papers/2407.20183" style="margin-right: 15px; text-decoration: none; color: #4A90E2;">📚 Hugging Face Papers</a>
        <a href="https://huggingface.co/spaces/internlm/MindSearch" style="text-decoration: none; color: #4A90E2;">🤗 Hugging Face Demo</a>
    </div>
    """)
    with gr.Row():
        with gr.Column(scale=10):
            with gr.Row():
                with gr.Column():
                    planner = gr.Chatbot(label='planner',
                                         height=700,
                                         show_label=True,
                                         show_copy_button=True,
                                         bubble_full_width=False,
                                         render_markdown=True)
                with gr.Column():
                    searcher = gr.Chatbot(label='searcher',
                                          height=700,
                                          show_label=True,
                                          show_copy_button=True,
                                          bubble_full_width=False,
                                          render_markdown=True)
            with gr.Row():
                user_input = gr.Textbox(show_label=False,
                                        placeholder='帮我搜索一下 InternLM 开源体系',
                                        lines=5,
                                        container=False)
            with gr.Row():
                with gr.Column(scale=2):
                    submitBtn = gr.Button('Submit')
                with gr.Column(scale=1, min_width=20):
                    emptyBtn = gr.Button('Clear History')

    def user(query, history):
        return '', history + [[query, '']]

    submitBtn.click(user, [user_input, planner], [user_input, planner],
                    queue=False).then(predict, [planner, searcher],
                                      [planner, searcher])
    emptyBtn.click(rst_mem, [planner, searcher], [planner, searcher],
                   queue=False)

demo.queue()
demo.launch(server_name='0.0.0.0',
            server_port=7860,
            inbrowser=True,
            share=True)
```

![image-20240822203640467](2-6_MindSearch.assets/image-20240822203640467.png)

4. 将 mindsearch_deploy 目录添加 git 

由于 studio 服务器连不上 hf, 因此将文件下载到本地电脑再上传

```sh
git clone https://huggingface.co/spaces/Yuki0943/MindSearchDemo

# 将代码放入 MindSearchDemo 中

git add .
git commit -m'init'
git push
```

![image-20240822205124006](2-6_MindSearch.assets/image-20240822205124006.png)

5. 启动

https://huggingface.co/spaces/Yuki0943/MindSearchDemo

问题: Z790主板和X670主板的区别是什么

![image-20240822211801810](2-6_MindSearch.assets/image-20240822211801810.png)
