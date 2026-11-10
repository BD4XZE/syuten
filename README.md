# syuten

面向考研英语的**词义消歧**命令行工具与 Rust 库。

给定一个英文单词，`syuten` 会从内置词典中取出它的全部中文释义，从内置真题语料中检索该词的例句，
再用一个重排序模型（reranker）对“例句 × 释义”逐对打分，最后按出现频次输出最可能的中文释义。

- 模型：`BAAI/bge-reranker-large`（XLM-RoBERTa-large 结构，隐藏层 1024，24 层）
- 推理后端：`candle` + CUDA（需要可用的 NVIDIA GPU）
- 词典与例句数据在**编译期嵌入二进制**，运行时不依赖数据文件

---

## 目录

- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
  - [构建](#构建)
  - [模型获取](#模型获取)
- [命令行用法](#命令行用法)
- [作为 Rust 库使用](#作为-rust-库使用)
- [作为 Python 扩展使用](#作为-python-扩展使用)
- [环境变量](#环境变量)
- [目录结构](#目录结构)
- [数据说明](#数据说明)
- [性能说明](#性能说明)
- [许可证](#许可证)

---

## 功能特性

- **中文释义消歧**：对同一单词的多个中文释义按语境打分排序。
- **真题例句检索**：从内置考研英语语料中按词形匹配例句，并高亮目标词前置以提升打分准确度。
- **频次统计**：汇总每条例句的最优释义，输出出现频率最高的前 5 项。
- **交互模式**：无参数运行时进入 REPL，一次加载模型、连续查询。
- **多种集成方式**：可直接作为 CLI 使用，也可作为 Rust 库或 Python 扩展调用。
- **零数据依赖**：词典与例句编译进二进制，发布只需分发可执行文件。

## 环境要求

| 依赖 | 说明 |
| --- | --- |
| Rust | 使用 edition 2024，要求 `rustc >= 1.85` |
| CUDA | `cudarc` 启用了 `cuda-13030` 特性，对应 CUDA 13.0.3 工具链 |
| GPU | 需要可用的 NVIDIA GPU（代码固定使用 `cuda(0)` 设备） |
| 磁盘 | 首次运行需下载约 2.2 GB 模型权重 |

## 快速开始

### 构建

```bash
# 开发构建
cargo build

# 发布构建（启用 strip 与 abort 优化）
cargo build --release
```

构建产物：

- 可执行文件：`target/<profile>/syuten`
- 库文件：`target/<profile>/libsyuten.{so,dylib}`（`cdylib`，供 Python 使用）
- 静态/常规库：`target/<profile>/libsyuten.rlib`（供 Rust 引用）

### 模型获取

程序按以下优先级查找模型目录（目录内需包含 `config.json`、`model.safetensors`、`tokenizer.json`）：

1. **环境变量 `SYUTEN_MODEL_DIR`**：直接指定本地模型目录，适用于离线环境或自备模型。
2. **自动下载**：未设置该变量时，从 HuggingFace 下载 `BAAI/bge-reranker-large`
   到标准缓存目录，并自动复用缓存。

```bash
# 方式一：离线 / 自备模型
export SYUTEN_MODEL_DIR=/path/to/model-dir
./target/release/syuten abandon

# 方式二：默认在线下载（缓存到 ~/.cache/huggingface）
./target/release/syuten abandon

# 方式三：使用镜像加速（国内推荐）
export HF_ENDPOINT=https://hf-mirror.com
./target/release/syuten abandon
```

模型缓存位置默认遵循 HuggingFace 约定，可通过 `HF_HOME` 修改。
下载后的目录形如：

```
$HF_HOME/hub/models--BAAI--bge-reranker-large/snapshots/<revision>/
├── config.json
├── model.safetensors
└── tokenizer.json
```

> 仓库中的 `src/model/` 是历史本地模型副本，已被 `.gitignore` 排除权重文件，不影响自动下载逻辑。

## 命令行用法

```text
syuten [WORD] [OPTIONS]
```

| 参数 | 说明 |
| --- | --- |
| `WORD` | 待查询的英文单词；省略时进入交互模式 |
| `-s, --sentences` | 在结果表格中额外输出使用到的真题例句 |
| `-d, --detail` | 预留参数（当前版本尚未参与输出逻辑） |

示例：

```bash
# 查询单个单词
syuten abandon

# 同时展示例句
syuten abandon -s

# 交互模式（Ctrl+D / Ctrl+C 退出）
syuten
```

输出为自适应终端宽度的表格，列为 `Freq`（频次）与 `Meaning`（中文释义），
加 `-s` 时追加 `Egsentence`（真题例句）列。

## 作为 Rust 库使用

```rust
use candle_core::Device;
use syuten::Reranker;
use syuten::model_source::resolve_model_dir;

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let model_dir = resolve_model_dir()?;
    let reranker = Reranker::new(model_dir, device)?;

    let sentence = "He decided to abandon the plan.";
    let meanings = vec!["放弃", "抛弃", "沉溺"];
    let scores = reranker.compute_scores(sentence, &meanings)?;

    println!("{:?}", scores);
    Ok(())
}
```

常用接口：

- `Reranker::new(model_dir, device)`：加载配置、权重与分词器（内部启用 512 最大长度截断）。
- `Reranker::compute_scores(sentence, meanings)`：返回每个释义的匹配得分。
- `syuten::get_max(meanings, scores, sentence)`：取最高分释义，返回 `CandidateScore`。
- `syuten::candle_model::CandleStrTool`：词典检索（`find_word_chinese_meaning`）与例句检索（`get_sentences`）。

## 作为 Python 扩展使用

库以 `cdylib` 形式编译，并导出 Python 模块 `libsyuten`（依赖 `pyo3` 的 `extension-module` 特性）。

```bash
cargo build --release
```

将生成的 `libsyuten.so`（macOS 为 `libsyuten.dylib`）所在目录加入 `PYTHONPATH` 后：

```python
import libsyuten

# 批量分析单词，结果追加写入 CSV（列为 meaning/index/text/score）
libsyuten.write_csv(["abandon", "abstract"], "result.csv")
```

> `write_csv` 以追加模式写入并禁用表头，重复调用会在同一文件中累加数据。

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `SYUTEN_MODEL_DIR` | 指定本地模型目录，设置后跳过下载 |
| `HF_HOME` | HuggingFace 缓存根目录（默认 `~/.cache/huggingface`） |
| `HF_ENDPOINT` | HuggingFace 服务端点，可设为镜像，如 `https://hf-mirror.com` |

## 目录结构

```text
.
├── Cargo.toml                 # 包定义与依赖
├── Cargo.lock
├── benches/
│   └── syuten.rs              # criterion 基准测试（manual_run）
└── src/
    ├── main.rs                # 可执行入口（cli_run；含批量分析辅助代码）
    ├── lib.rs                 # 库入口：Reranker 模型封装、数据嵌入、get_max
    ├── cli.rs                 # 参数解析、交互循环、结果频次统计
    ├── candle_model.rs        # 词典/例句检索、推理编排、CSV 批量输出
    ├── model_source.rs        # 模型目录解析与 HuggingFace 自动下载
    ├── python_script.rs       # Python 扩展模块（libsyuten.write_csv）
    ├── show.rs                # 终端表格渲染（自适应宽度）
    ├── data/                  # 词典与例句数据（编译期嵌入）
    ├── model/                 # 历史本地模型副本（权重已忽略）
    ├── log/                   # 错误日志
    ├── prompt/                # 提示词文本
    └── result/                # 分析结果输出样例
```

## 数据说明

以下数据通过 `include_str!` 在编译期嵌入二进制，运行时无需携带原始文件：

| 文件 | 用途 |
| --- | --- |
| `src/data/dict_extract.json` | 英汉词典（默认） |
| `src/data/onnx_dict_extract.json` | 词典（`candle_model` 实际使用的版本） |
| `src/data/CET4_part_sentences.csv` | CET-4 例句语料 |
| `src/data/PEGG_restoration.csv` | 近 25 年考研真题语料（含词形还原列） |
| `src/data/PEGG_frequency.csv` | 词频表（可执行文件内批量分析使用） |

例句检索逻辑：对输入单词构造词形正则，在语料的“词形还原”列中匹配，
命中后取同行原始句子作为候选。

## 性能说明

- 单次查询的主要耗时在模型加载（约占 96%），因此提供了交互模式以复用已加载的模型。
- `compute_scores` 采用批量编码并统一 padding；序列长度上限为 512（模型限制，超出会告警）。
- 基准测试：

```bash
cargo bench
```

## 许可证

本项目采用 [AGPL-3.0-only](https://www.gnu.org/licenses/agpl-3.0.html) 许可证。
