# 本章主要尝试了浦语大模型的demo

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

