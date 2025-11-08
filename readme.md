# nano-whisper
nano-whisper is an inference engine that supports cross-attention encoder-decoder model and benchmarking vLLM & CTranslate2 with non-redundant code

Now available for whisper

At present, the whole project does not rely on any interfaces and data structures of vLLM, and the code size is only 2000+

## 引擎性能
#### 测试条件
1. [✅] 0.8B
2. [✅] 音频时长6s
3. [✅] mel编码时长以10ms计
4. [✅] A100单卡测试

| requests | 1       | 16      | 32      | 64      | 128     | 256     |
|----------|---------|---------|---------|---------|---------|---------|
| CTranslate2 | 0.130/0.130 | -/0.130 | -/0.130 | -/0.130 | -/0.130 | -/0.130 |
| vLLM | 0.205/0.205 | 2.215/0.138 | 2.488/0.077 | 4.708/0.074 | 6.048/0.047 | 8.624/0.033 |
| nanowhisper eager | 0.190/0.190 | 1.854/0.115 | 2.347/0.073 | 3.394/0.053 | 4.705/0.036 | 7.630/0.030 |
| nanowhisper cuda graph | 0.090/0.090 | 1.113/0.069 | 0.867/0.027 | 2.702/0.042 | 4.983/0.039 | 8.192/0.032 |

*单位：seconds*

#### 不同数量请求预热效果（主要原因为部分显存为torch管理）

##### 1 请求预热

| requests | 1       | 16      | 32      | 64      | 128     | 256     |
|----------|---------|---------|---------|---------|---------|---------|
| nanowhisper eager | 0.190/0.190 | 1.854/0.115 | 2.347/0.073 | 3.394/0.053 | 4.705/0.036 | 7.630/0.030 |
| nanowhisper cuda graph | 0.090/0.090 | 1.113/0.069 | 0.867/0.027 | 2.702/0.042 | 4.983/0.039 | 8.192/0.032 |

*单位：seconds*

##### 256 请求预热

| requests | 1       | 16      | 32      | 64      | 128     | 256     |
|----------|---------|---------|---------|---------|---------|---------|
| nanowhisper eager | 0.195/0.195 | 1.367/0.085 | 1.196/0.037 | 2.735/0.043 | 3.728/0.029 | 6.772/0.026 |
| nanowhisper cuda graph | 0.084/0.084 | 1.106/0.069 | 0.861/0.027 | 2.705/0.042 | 3.708/0.028 | 6.985/0.027 |

*单位：seconds*
