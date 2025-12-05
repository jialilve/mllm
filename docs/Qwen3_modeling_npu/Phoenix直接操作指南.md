# 在 Phoenix 上直接操作指南

## 概述

**推荐方式**：直接在 Phoenix 服务器上通过 adb 连接手机并推送文件，无需先复制到主要开发服务器。这样更高效，因为模型文件已经在 Phoenix 上了。

## 前提条件检查

### 1. 检查 Phoenix 上是否有 adb

```bash
# SSH 连接到 Phoenix
ssh -p 20212 zhu_lei@10.109.246.210

# 检查 adb 是否可用
which adb
adb version
```

### 2. 检查网络连通性

```bash
# 在 Phoenix 上测试能否访问 Android 设备
ping 10.29.208.59

# 测试 adb 端口
telnet 10.29.208.59 9808
# 或
nc -zv 10.29.208.59 9808
```

### 3. 如果 adb 不存在，安装 Android SDK Platform Tools

```bash
# 方法 1: 使用包管理器（如果可用）
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install android-tools-adb android-tools-fastboot

# CentOS/RHEL
sudo yum install android-tools

# 方法 2: 手动下载安装
# 下载 Android SDK Platform Tools
wget https://dl.google.com/android/repository/platform-tools-latest-linux.zip
unzip platform-tools-latest-linux.zip
export PATH=$PATH:$(pwd)/platform-tools
```

## 完整操作流程

### 步骤 1: 在 Phoenix 上编译代码（如果有编译环境）

```bash
# SSH 连接到 Phoenix
ssh -p 20212 zhu_lei@10.109.246.210

# 进入项目目录
cd /data/shrelic/mllm_v2

# 检查代码是否存在
ls -la

# 编译项目
python task.py tasks/build_android_qnn_debug.yaml

# 或使用 CMake
cd build-android-qnn-dbg
cmake .. -DCMAKE_BUILD_TYPE=Debug -DMLLM_BUILD_QNN_BACKEND=ON
make -j$(nproc) mllm-qwen3-npu

# 检查编译产物
ls -lh build-android-qnn-dbg/bin/mllm-qwen3-npu
```

**注意**：如果 Phoenix 上没有编译环境，可以在主要开发服务器上编译，然后通过 scp 将编译产物复制到 Phoenix：

```bash
# 在主要开发服务器上编译后
scp -P 20212 ~/mllm_v2/build-android-qnn-dbg/bin/mllm-qwen3-npu \
    zhu_lei@10.109.246.210:/data/shrelic/mllm_v2/build-android-qnn-dbg/bin/

# 推送所有库文件
scp -P 20212 ~/mllm_v2/build-android-qnn-dbg/bin/lib*.so \
    zhu_lei@10.109.246.210:/data/shrelic/mllm_v2/build-android-qnn-dbg/bin/
```

### 步骤 2: 在 Phoenix 上连接 Android 设备

```bash
# 在 Phoenix 上
adb connect 10.29.208.59:9808

# 检查连接状态
adb devices

# 应该看到类似输出：
# List of devices attached
# 10.29.208.59:9808    device
```

### 步骤 3: 在 Phoenix 上推送文件到 Android 设备

```bash
# 在 Phoenix 上
cd /data/shrelic/mllm_v2/build-android-qnn-dbg/bin

# 推送可执行文件和库文件
adb push ./mllm-qwen3-npu /data/local/tmp/zl/mllm-v2/bin_PR/
adb push ./libMllmQNNBackend.so /data/local/tmp/zl/mllm-v2/bin_PR/
adb push ./libCustomPackageForHostTest.so /data/local/tmp/zl/mllm-v2/bin_PR/
adb push ./libMllmCPUBackend.so /data/local/tmp/zl/mllm-v2/bin_PR/
adb push ./libMllmRT.so /data/local/tmp/zl/mllm-v2/bin_PR/
adb push ./MllmFFIExtension.so /data/local/tmp/zl/mllm-v2/bin_PR/

# 推送模型文件（模型文件已经在 Phoenix 上，直接推送）
adb push /data/shrelic/mllm_v2/qwen3-1.7b-int8-rotated.mllm \
    /data/local/tmp/zl/mllm-v2/bin_PR/qwen3-model.mllm

# 推送配置文件（如果有）
adb push /data/shrelic/mllm_v2/config.json \
    /data/local/tmp/zl/mllm-v2/bin_PR/config.json

# 推送 tokenizer（如果有）
adb push /data/shrelic/mllm_v2/tokenizer.json \
    /data/local/tmp/zl/mllm-v2/bin_PR/tokenizer.json
```

**优势**：模型文件已经在 Phoenix 上，不需要先复制到主要开发服务器，直接推送即可！

### 步骤 4: 在 Android 设备上运行

```bash
# 在 Phoenix 上通过 adb shell 进入设备
adb shell

# 在设备 shell 中
cd /data/local/tmp/zl/mllm-v2/bin_PR
export LD_LIBRARY_PATH=.
export ADSP_LIBRARY_PATH=.  # 如果需要

# 运行程序
./mllm-qwen3-npu
```

## 常见问题

### Q1: Phoenix 上找不到 adb 命令

**解决方案**：
1. 检查是否已安装：`which adb`
2. 如果没有，安装 Android SDK Platform Tools（见前提条件检查）
3. 或者使用完整路径：`/path/to/platform-tools/adb`

### Q2: adb connect 失败，无法连接设备

**可能原因**：
1. 网络不通：Phoenix 无法访问 Android 设备的网络
2. 防火墙阻止：端口 9808 被防火墙阻止
3. adb server 问题：尝试重启 adb server

**解决方案**：
```bash
# 重启 adb server
adb kill-server
adb start-server
adb connect 10.29.208.59:9808

# 检查网络连通性
ping 10.29.208.59
telnet 10.29.208.59 9808
```

### Q3: 如果 Phoenix 无法直接连接手机怎么办？

**解决方案**：使用方式 B（通过主要开发服务器中转），见 `完整工作流程.md` 中的方式 B。

### Q4: 模型文件推送很慢

**解决方案**：
- 模型文件通常很大（几 GB），推送需要时间，这是正常的
- 可以使用 `adb push` 的进度显示查看状态
- 确保网络连接稳定

## 优势总结

✅ **更高效**：模型文件已经在 Phoenix 上，不需要额外传输  
✅ **更直接**：减少文件传输步骤  
✅ **更简单**：在一个地方完成所有操作  

## 工作流程对比

### 方式 A（推荐）：在 Phoenix 上直接操作
```
Phoenix 服务器
├── 模型文件（已存在）
├── 编译代码（或从主要开发服务器复制编译产物）
├── adb connect 手机
└── adb push 文件到手机
```

### 方式 B：通过主要开发服务器中转
```
主要开发服务器
├── 编译代码
├── 从 Phoenix scp 获取模型文件
├── adb connect 手机
└── adb push 文件到手机
```

**推荐优先尝试方式 A**，如果网络不通或没有 adb，再使用方式 B。

