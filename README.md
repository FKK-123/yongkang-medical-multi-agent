# chenguang-agent-

#### 介绍
《永康医疗-智能体》集中于解决医疗领域的 Agent 业务构建。包括：
1. 智慧问诊 Agent
2. 报告解读 Agent
3. 药物 Agent
4. 知识文档 Agent
5. 运营数据 Agent

#### 软件架构
软件架构说明


#### 安装教程

```sh
# 创建环境
conda create -n yongkang python=3.13
# 激活环境
conda activate yongkang

# 安装依赖
pip install -r requirements.txt
```


## 指定端口启动
```
uvicorn src.main:app --port 8080 --reload
```

## env配置
把 .env.example 复制一份 叫 .env ，修改为自己的信息即可

