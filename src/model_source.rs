//! 模型目录解析与自动下载
//!
//! 解析优先级：
//! 1. 环境变量 `SYUTEN_MODEL_DIR`：直接使用本地已有的模型目录，
//!    目录内需包含 config.json / model.safetensors / tokenizer.json
//! 2. 从 HuggingFace 自动下载默认模型到标准缓存目录
//!    （支持 `HF_HOME` 自定义缓存位置、`HF_ENDPOINT` 设置镜像，如 https://hf-mirror.com）
use anyhow::{Context, Result, ensure};
use std::path::PathBuf;

/// 默认模型仓库
pub const DEFAULT_MODEL_REPO: &str = "BAAI/bge-reranker-large";

/// 解析模型目录：优先环境变量指定的本地目录，否则从 HuggingFace 下载并返回缓存目录
pub fn resolve_model_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("SYUTEN_MODEL_DIR") {
        let dir = PathBuf::from(dir);
        ensure!(
            dir.is_dir(),
            "SYUTEN_MODEL_DIR 指向的路径不存在或不是目录: {:?}",
            dir
        );
        return Ok(dir);
    }
    download_model_dir()
}

/// 从 HuggingFace 下载模型到标准缓存目录，返回缓存中的快照目录
/// 三个文件都位于同一个快照目录，该目录即模型目录
fn download_model_dir() -> Result<PathBuf> {
    let api = hf_hub::api::sync::Api::new().context("初始化 HuggingFace 客户端失败")?;
    let repo = api.model(DEFAULT_MODEL_REPO.to_string());

    // 先下载小文件快速失败，最后下载约 2.2GB 的权重文件
    repo.get("config.json")
        .context("下载 config.json 失败")?;
    repo.get("tokenizer.json")
        .context("下载 tokenizer.json 失败")?;
    let weights_path = repo
        .get("model.safetensors")
        .context("下载 model.safetensors 失败（约 2.2GB，首次下载请耐心等待）")?;

    Ok(weights_path
        .parent()
        .context("模型缓存快照目录异常")?
        .to_path_buf())
}
