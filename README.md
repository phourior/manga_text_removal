# Demo

## 环境

**以下为开发平台配置：**

* 20/30系显卡

* pytorch版本：2.1.1+cu121

## 安装

1. 安装python

1. 安装pytorch(cuda版本)

2. `pip install -r requirements.txt`(没测试过)

3. 模型位置：

   *model*
   |--*mask2former*
   |  |--**config.json**
   |  |--**model.safetensors**
   |
   |--**big-lama.pt**

## 运行

1. `python demo.py`

* **注意：每次启动时模型初始化需要一段时间，此时不会创建图形界面**

## 使用

选择图片——生成mask——画笔修改mask——生成修复图像——保存修复图像

*连⑨都会用*

* **注意：别去测试非法输入，没做处理也不准备做**

