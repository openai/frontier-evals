# PaperBench Gemini 2.5 Pro Judge 实施方案

## 项目概述

本文档描述了在PaperBench评测系统中添加Google Gemini 2.5 Pro模型支持的技术实施方案。通过适配器模式，我们可以在保持现有OpenAI基础设施不变的情况下，支持使用Gemini 2.5 Pro进行论文复现评测。

## 背景分析

### 当前架构
- PaperBench使用`SimpleJudge`进行评测
- 评测系统基于`TurnCompleter`抽象接口
- 默认使用OpenAI的`o3-mini-2025-01-31`模型
- 支持`reasoning_effort="high"`参数（OpenAI专有特性）

### 核心挑战
1. **OpenAI专有特性**：`reasoning_effort`参数Gemini不支持
2. **类型系统耦合**：深度依赖OpenAI的类型定义
3. **Token计算**：使用tiktoken（OpenAI专用）
4. **Docker环境**：需要配置Google API访问

### 解决策略
采用**适配器模式**，创建`GoogleCompletionsTurnCompleter`类，让Gemini API"伪装"成OpenAI接口，最小化对现有代码的修改。**专门支持Gemini 2.5 Pro，具有2M tokens的超大上下文窗口。**

## Gemini 2.5 Pro 特性

### 模型优势
- **超大上下文窗口**：2,000,000 tokens（远超OpenAI模型）
- **优秀的推理能力**：适合复杂的代码评测任务
- **成本效益**：相比OpenAI模型更经济
- **多模态支持**：未来可扩展支持图像等

### 配置默认值
- **模型**：`gemini-2.5-pro`
- **温度**：`0.1`（保证评测一致性）
- **上下文窗口**：`2,000,000` tokens

## 技术方案

### 1. 核心组件设计

#### 1.1 GoogleCompletionsTurnCompleter类

创建新文件：`preparedness_turn_completer/google_completions_turn_completer.py`

**关键特性：**
- 继承自`TurnCompleter`基类
- 实现与OpenAI完全兼容的接口
- 内部转换OpenAI消息格式到Google格式
- 返回标准的OpenAI格式响应
- 忽略`reasoning_effort`等OpenAI专有参数

**核心方法：**
```python
class GoogleCompletionsTurnCompleter(TurnCompleter):
    async def async_completion(self, conversation, **params) -> TurnCompleter.Completion
    def _convert_to_google_format(self, messages) -> str
    def _estimate_tokens(self, messages) -> int
    def _get_google_context_window(self, model: str) -> int
```

#### 1.2 配置系统扩展

修改`paperbench/nano/structs.py`中的`JudgeConfig`类：

**新增字段：**
- `use_google: bool = False` - 启用Gemini开关
- `google_api_key: str | None = None` - Google API密钥

**自动配置逻辑：**
- 当`use_google=True`时，自动替换`completer_config`为Google配置
- 自动更新Docker环境变量包含`GOOGLE_API_KEY`
- 保持向后兼容，默认仍使用OpenAI

### 2. 消息格式转换

#### 2.1 OpenAI到Google转换
```python
def _convert_to_google_format(self, messages):
    """
    OpenAI: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    Google: "System: ...\nUser: ...\n"
    """
```

### 3. SimpleJudge 解析流程调整（仅Gemini）

在仅使用Gemini的场景，无法依赖OpenAI的`response_format`保证严格JSON输出。因此做如下调整：

- 复用主`completer_config`（即Google/Gemini）作为解析用的completer，避免对OpenAI的硬依赖；
- 强化解析步骤的系统提示：要求“仅输出严格JSON（无额外文本、无markdown代码块）”；
- 增加JSON提取fallback：从模型输出文本中提取首个JSON对象再解析，提升鲁棒性；
- 若仍解析失败，将抛出解析异常并在上层处理为该叶节点的解析错误。

对应代码改动：`paperbench/judge/simple.py`
- `_init_structured_completer`：当`self.completer_config`不是OpenAI配置时，复用当前completer；否则仍使用OpenAI并传入`response_format`。
- `_parse_model_response`：强化系统提示，并新增`_extract_json_object`辅助以从自由文本提取JSON。

#### 2.2 Google到OpenAI转换
```python
def _convert_google_response(self, response):
    """
    Google response -> ChatCompletionMessage格式
    """
```

### 3. Token计算策略

**问题：** Gemini不提供详细的token使用统计

**解决方案：**
- 使用基于字符数的估算：`tokens ≈ characters / 4`
- 足够准确用于context window管理
- 保持与现有token统计系统的兼容性

### 4. Docker环境配置

#### 4.1 依赖管理
更新`pyproject.toml`：
```toml
dependencies = [
    # ... 现有依赖
    "google-generativeai>=0.5.0",
]
```

#### 4.2 Docker镜像
更新`paperbench/Dockerfile.base`：
```dockerfile
RUN /opt/conda/envs/${CONDA_ENV_NAME}/bin/pip install google-generativeai
```

#### 4.3 环境变量
自动在Docker容器中设置：
```python
environment={"GOOGLE_API_KEY": google_api_key}
```

### 5. 使用方式

#### 5.1 命令行配置
```bash
# 使用Gemini 2.5 Pro评测
uv run python -m paperbench.nano.entrypoint \
    paperbench.judge.use_gemini=true \
    paperbench.judge.gemini_api_key=${GOOGLE_API_KEY} \
    # 其他参数保持不变

# 或者设置环境变量后简化命令
export GOOGLE_API_KEY="your-google-api-key"
uv run python -m paperbench.nano.entrypoint \
    paperbench.judge.use_gemini=true
```

#### 5.2 环境变量
```bash
# 主要API密钥
export GOOGLE_API_KEY="your-google-api-key"

# 可选：专门的评测用密钥
export GRADER_GOOGLE_API_KEY="your-grader-specific-key"
```

## 实施计划

### Phase 1: 核心实现 (3-5天)
- [ ] 创建`GoogleCompletionsTurnCompleter`类
- [ ] 实现消息格式转换逻辑
- [ ] 添加Google模型context window配置
- [ ] 实现token估算机制

### Phase 2: 集成测试 (2-3天)
- [ ] 修改`JudgeConfig`支持Google配置
- [ ] 更新依赖管理
- [ ] 修改Docker镜像
- [ ] 端到端功能测试

### Phase 3: 完善和文档 (1-2天)
- [ ] 错误处理和重试机制
- [ ] 用户文档和使用示例
- [ ] 性能优化

**总工作量：1-1.5周**

## 技术细节

### 支持的Gemini模型
```python
SUPPORTED_GOOGLE_MODELS = {
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-pro-002": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
}
```

### 参数映射
| OpenAI参数 | Gemini参数 | 处理方式 |
|------------|------------|----------|
| `model` | `model` | 直接映射 |
| `temperature` | `temperature` | 直接映射 |
| `max_tokens` | `max_output_tokens` | 直接映射 |
| `reasoning_effort` | - | **忽略** |
| `response_format` | - | 目前忽略 |
| `tools` | - | 未来可扩展 |

### 错误处理
- Google API异常映射到通用异常
- 网络重试机制
- API限制处理

## 风险评估

### 低风险
- ✅ 不修改现有OpenAI workflow
- ✅ 增量添加，可快速回退
- ✅ 保持完全向后兼容

### 中等风险
- ⚠️ Token估算精度（可接受）
- ⚠️ 新的API依赖
- ⚠️ 额外的配置复杂度

### 缓解措施
- 全面的单元测试和集成测试
- 详细的错误日志和监控
- 清晰的配置文档

## 测试策略

### 单元测试
- GoogleCompletionsTurnCompleter各方法
- 消息格式转换正确性
- Token估算准确性
- 配置验证逻辑

### 集成测试
- 端到端评测流程
- Docker容器环境
- 与现有OpenAI系统并行测试

### 性能测试
- API响应时间
- Token使用效率
- 内存和CPU使用

## 维护考虑

### 长期维护
- Google API版本更新
- 模型列表维护
- 性能监控和优化

### 扩展性
- 支持更多Google模型参数
- 添加工具调用支持
- 结构化输出支持

## 结论

通过适配器模式，我们可以以最小的修改成本在PaperBench中支持Gemini评测。该方案：

- **技术可行**：基于成熟的抽象接口
- **风险可控**：不影响现有功能
- **工作量合理**：1-1.5周可完成
- **易于维护**：代码简洁，逻辑清晰

关键是忽略OpenAI专有的`reasoning_effort`参数，这使得整个实施方案大幅简化，从原本需要4-6周的大型重构降低到1-1.5周的适配工作。
