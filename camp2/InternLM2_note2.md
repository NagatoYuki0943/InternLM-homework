本章主要尝试了浦语大模型的demo，因此笔记不多，这里主要记录的自己实验过程中的注意点

## 部署 `浦语·灵笔2` 模型出现的问题

在使用教程中如下代码进行部署模型时会发生无法从 `examples/utils.py` 获取对应函数的问题，可能是由于我自己建立环境的原因，解决办法也很简单，不从 `examples/utils.py` 中获取函数，直接把对应的函数和导入的包直接复制到 `gradio_demo_composition.py` 和 `gradio_demo_chat.py`中即可顺利运行。 

```sh
cd /root/demo/InternLM-XComposer
python /root/demo/InternLM-XComposer/examples/gradio_demo_composition.py  \
--code_path /root/models/internlm-xcomposer2-7b \
--private \
--num_gpus 1 \
--port 6006

cd /root/demo/InternLM-XComposer
python /root/demo/InternLM-XComposer/examples/gradio_demo_chat.py  \
--code_path /root/models/internlm-xcomposer2-vl-7b \
--private \
--num_gpus 1 \
--port 6006
```

`gradio_demo_composition.py` 和 `gradio_demo_chat.py` 的代码中有把上级和本级目录添加到环境变量的代码，不过生效似乎不完整，不是很理解。可以找到上级目录中 `demo_asset` 的文件夹代码，但没法找到 `examples` 目录下的代码

```sh
import os
import re
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
```

## 将服务器ip映射到本地ip的方法

```sh
# 从本地使用 ssh 连接 studio 端口
# 6006 是要映射的ip
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374

# 映射之后在本地浏览器即可访问
http://127.0.0.1:6006 
```

## 软链接

```sh
# 建立软链接
ln -s source link_name
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b

# 删除软链接
unlink link_name
unlink /root/models/internlm2-chat-7b
```

## Terminal 终端清除方法

之前使用Jupter开启了终端运行模型，后来想暂停终端时尝试使用 `Ctrl + C` 终止，不过响应很慢，就把终端关掉了，但是发现显卡上仍然有模型，当时的解决办法时直接关闭虚拟机完成的，现在才明白可以在Jupyer左侧强制终止终端。

