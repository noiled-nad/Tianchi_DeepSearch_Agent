# V2Ray 代理服务使用与配置指南

## 目录
1. [简介](#简介)
2. [服务配置](#服务配置)
3. [使用方法](#使用方法)
4. [修改配置](#修改配置)
5. [故障排查](#故障排查)

## 简介

本文档介绍如何在阿里云DSW环境中使用已配置的V2Ray代理服务，包括服务的基本信息、使用方法以及如何修改配置。

## 服务配置

### 入站代理配置
- **HTTP代理端口**: `1080`
- **SOCKS代理端口**: `1081`
- **监听地址**: `0.0.0.0` (允许所有接口访问)
- **认证**: 无认证

### 出站代理配置
- **协议**: ShadowSocks (SS)
- **加密方式**: `chacha20-ietf-poly1305`
- **密码**: `9d9a6fa0-0fb7-4bc7-bd2e-da41d84cf965`

### 路由规则
1. 特定域名通过代理:
   - `serpapi.com`
   - `google.serper.dev`
   - 国际网站: Google, YouTube, Facebook, Twitter, Instagram, Netflix, Disney, Spotify 等
2. 私有IP地址: 阻止访问
3. 广告域名: 阻止访问
4. 其他流量: 直连

## 使用方法

### 1. 在DSW环境中使用代理

在DSW环境中，您可以使用以下命令通过代理访问网络：

```bash
# 通过HTTP代理访问网站
curl --proxy http://127.0.0.1:1080 https://www.google.com

# 通过SOCKS代理访问网站
curl --socks5 127.0.0.1:1081 https://www.google.com
```

### 2. 在本地设备上使用代理

要在本地设备上使用此代理服务：

1. 在DSW界面中点击"Port"按钮
2. 找到端口`1080`或`1081`，点击"Open Port"
3. 系统会生成一个URL，例如: `https://dsw-gateway-cn-shanghai.data.aliyun.com/dsw-697167/ide/proxy/1080/`
4. 在本地设备的浏览器或应用程序中配置代理:
   - HTTP代理: 使用生成的URL和端口1080
   - SOCKS代理: 使用生成的URL和端口1081

### 3. 在Python代码中使用代理

```python
import os
import requests

# 设置环境变量使用代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1080'

# 或者在requests中直接指定
proxies = {
    'http': 'http://127.0.0.1:1080',
    'https': 'http://127.0.0.1:1080'
}

response = requests.get('https://www.google.com', proxies=proxies)
```

## 修改配置

### 1. 修改出站服务器信息

如果您需要更换SS服务器，请按以下步骤操作：

1. **停止当前V2Ray服务**：
   ```bash
   pkill v2ray
   ```

2. **编辑配置文件**：
   ```bash
   nano /etc/v2ray/config.json
   ```

3. **修改出站服务器配置**：
   在 `outbounds` 部分找到 ShadowSocks 配置：
   ```json
   {
     "protocol": "shadowsocks",
     "settings": {
       "servers": [{
         "address": "beinga.ewpbh.cn",      // <- 修改服务器地址
         "port": 33001,                    // <- 修改端口
         "method": "chacha20-ietf-poly1305", // <- 修改加密方式
         "password": "9d9a6fa0-0fb7-4bc7-bd2e-da41d84cf965"  // <- 修改密码
       }]
     },
     "tag": "hk1-ss"
   }
   ```

4. **测试配置文件**：
   ```bash
   v2ray test -c /etc/v2ray/config.json
   ```

5. **重启V2Ray服务**：
   ```bash
   v2ray run -c /etc/v2ray/config.json &
   ```

### 2. 修改路由规则

要修改路由规则，编辑 `/etc/v2ray/config.json` 中的 `routing.rules` 部分：

```json
"rules": [
  // 特定域名通过代理
  {
    "type": "field",
    "network": "tcp,udp",
    "domain": [
      "serpapi.com",
      "google.serper.dev"
    ],
    "outboundTag": "hk1-ss"
  },
  // 国际网站通过代理
  {
    "type": "field",
    "network": "tcp,udp",
    "domain": [
      "geosite:google",
      "geosite:youtube",
      "geosite:facebook"
    ],
    "outboundTag": "hk1-ss"
  },
  // 其他流量直连
  {
    "type": "field",
    "network": "tcp,udp",
    "outboundTag": "direct"
  }
]
```

**注意**：路由规则按照数组顺序匹配，一旦匹配到规则就执行对应操作，不再继续匹配后续规则。

### 3. 添加新的代理服务器

如果要添加新的服务器并实现负载均衡或故障转移：

1. 在 `outbounds` 部分添加新的出站配置：
   ```json
   {
     "protocol": "shadowsocks",
     "settings": {
       "servers": [{
         "address": "new-server.com",
         "port": 12345,
         "method": "chacha20-ietf-poly1305",
         "password": "new_password"
       }]
     },
     "tag": "new-ss"
   }
   ```

2. 在路由规则中使用新的出站标签：
   ```json
   {
     "type": "field",
     "network": "tcp,udp",
     "domain": [
       "example.com"
     ],
     "outboundTag": "new-ss"
   }
   ```

## 故障排查

### 1. 测试配置文件

在启动服务前，始终测试配置文件：
```bash
v2ray test -c /etc/v2ray/config.json
```

### 2. 检查服务状态

检查V2Ray是否正在运行：
```bash
ps aux | grep v2ray
```

### 3. 重启服务

如果出现问题，可以重启服务：
```bash
pkill v2ray
v2ray run -c /etc/v2ray/config.json &
```

### 4. 测试代理连接

测试代理是否正常工作：
```bash
curl --proxy http://127.0.0.1:1080 https://httpbin.org/ip
```

### 5. 查看日志

V2Ray会在终端输出日志信息，显示流量通过哪个出站代理：
- `[hk1-ss]` 表示流量通过香港1服务器
- `[direct]` 表示流量直连
- `[blocked]` 表示流量被阻止

### 6. 常见问题

- **无法连接到代理端口**: 检查V2Ray服务是否正在运行
- **流量未按预期路由**: 检查路由规则的顺序和条件
- **配置文件错误**: 使用 `v2ray test` 命令验证配置文件
- **连接超时**: 检查SS服务器信息是否正确，网络是否可达

---

**注意**: 所有配置修改都需要重启V2Ray服务才能生效。在生产环境中，建议先在测试环境中验证配置。