pub mod candle_model;
pub mod cli;
pub mod model_source;
pub mod python_script;
pub mod show;
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};
use candle_transformers::models::bert::{BertModel, Config};
use serde::Serialize;
pub(crate) use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use tokenizers::{Tokenizer, TruncationParams};

//数据在编译期直接嵌入二进制,发布时无需携带数据文件.
pub(crate) const DICT_JSON: &str = include_str!("data/dict_extract.json");
pub(crate) const CET4_SENTENCES: &str = include_str!("data/CET4_part_sentences.csv");
pub(crate) const PEGG_SENTENCES: &str = include_str!("data/PEGG_restoration.csv");
pub(crate) type Dictionary = HashMap<String, Vec<String>>;

#[derive(Debug, Serialize, PartialEq,Clone)]
pub struct CandidateScore {
    pub index: usize,
    pub meaning: String,
    pub score: f32,
    // 原始的真题例句
    pub sentence: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ResourceError {
    DictNotFound,
    SentencesNotFound,
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DictNotFound => write!(f, "Word not in the dict"),
            Self::SentencesNotFound => write!(f, "未找到该词的例句",),
        }
    }
}

///输出排序，把输出塞到struct CandidateScore
pub fn get_max(
    meanings: &Vec<String>,
    scores: &Vec<f32>,
    sentence: String,
) -> Result<CandidateScore> {
    let (index, max_score) = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, &val)| (idx, val))
        .context("Get index and max value error")?;
    let meaning = meanings.get(index).context("sentence get error")?;
    let res = CandidateScore {
        index: index,
        meaning: meaning.clone(),
        score: max_score,
        sentence,
    };
    Ok(res)
}

/// 选择推理设备：优先使用 CUDA(0)，不可用时自动降级到 CPU。
pub fn select_device() -> Device {
    match Device::new_cuda(0) {
        Ok(device) => device,
        Err(err) => {
            eprintln!("[WARN] CUDA 不可用，已降级到 CPU: {err}");
            Device::Cpu
        }
    }
}

///加载大模型，并配置它
pub struct Reranker {
    model: BertModel,
    dense: Linear,
    out_proj: Linear,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

/// Load model
/// 这个模型加载部分代码由AI负责，我实在是不了解这些内容。
/// 如果你比较了解这部分内容，想要改进，欢迎提交issue。
impl Reranker {
    /// 从模型目录加载配置、权重与 Tokenizer
    pub fn new<P: AsRef<Path>>(model_dir: P, device: Device) -> Result<Self> {
        // 这里的环境变量必须是512,模型限制，没办法。
        // 如果想要更多的输入，需要更换模型，但是代价是性能，平时也与不到这么大的输入。
        // 遇到的单词都比较复杂。
        const MAX_SEQ_LEN: usize = 512;
        let dir = model_dir.as_ref();
        let config_path = dir.join("config.json");
        let model_path = dir.join("model.safetensors");
        let tokenizer_path = dir.join("tokenizer.json");

        // 1. 读取 Config
        let config_file = std::fs::File::open(&config_path)
            .with_context(|| format!("failed to open {:?}", config_path))?;
        let mut config: Config = serde_json::from_reader(config_file)?;
        config.model_type = Some("roberta".to_string());

        // 2. 加载模型权重
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };

        let model = BertModel::load(vb.clone(), &config)?;
        let dense = linear(
            config.hidden_size,
            config.hidden_size,
            vb.pp("classifier.dense"),
        )?;
        let out_proj = linear(config.hidden_size, 1, vb.pp("classifier.out_proj"))?;

        // 3. 加载 Tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        let mut tokenizer = tokenizer;

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQ_LEN,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("failed to enable truncation: {e}"))?;

        // comment for debug
        // println!("truncation enabled");
        // println!("truncation max_length = {}", MAX_SEQ_LEN);
        // if let Some(trunc) = tokenizer.get_truncation() {
        //     println!("tokenizer truncation config = {:?}", trunc);
        // } else {
        //     println!("[WARN] tokenizer truncation is NOT enabled!");
        // }

        Ok(Self {
            model,
            dense,
            out_proj,
            tokenizer,
            device,
        })
    }

    /// 对单个 Sentence 与多个 meanings 组合进行批量计算打分
    pub fn compute_scores(&self, sentence: &str, meanings: &[impl AsRef<str>]) -> Result<Vec<f32>> {
        ensure!(
            !meanings.is_empty(),
            "word meanings is empty in the dict dataset"
        );

        // eprintln!("[DEBUG] 输入句子长度: {} 字符 | 待评估释义数量: {}", sentence.len(), meanings.len());

        // 1. 批量分词
        let inputs: Vec<tokenizers::EncodeInput> = meanings
            .iter()
            .map(|doc| tokenizers::EncodeInput::Dual(sentence.into(), doc.as_ref().into()))
            .collect();

        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // 2. 维度与 Token 越界排查
        let batch_size = encodings.len();
        let mut max_len = 0;
        let mut max_token_id = 0u32;
        let mut max_type_id = 0u32;

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let type_ids = enc.get_type_ids();

            if ids.len() > max_len {
                max_len = ids.len();
            }
            for &id in ids {
                if id > max_token_id {
                    max_token_id = id;
                }
            }
            for &tid in type_ids {
                if tid > max_type_id {
                    max_type_id = tid;
                }
            }

            // 警告提示：超过 512 长度会直接触发 CUDA 崩溃
            if ids.len() > 512 {
                eprintln!(
                    "[WARN] 样本 #{} 长度超过 512 上限！实际 Token 数: {}",
                    i,
                    ids.len()
                );
            }
        }

        // eprintln!(
        //     "[DEBUG] Batch 大小: {} | 最大序列长度: {} | 最大 Token ID: {} | Tokenizer 原生最大 Type ID: {}",
        //     batch_size, max_len, max_token_id, max_type_id
        // );

        // 3. 构建 Flatten 数据并填充 Padding
        let mut input_ids_flat = Vec::with_capacity(batch_size * max_len);

        for enc in &encodings {
            let ids = enc.get_ids();
            let pad_len = max_len - ids.len();

            input_ids_flat.extend_from_slice(ids);
            input_ids_flat.extend(std::iter::repeat(0).take(pad_len));
        }

        // 4. 张量维度验证
        let input_ids = Tensor::from_slice(&input_ids_flat, (batch_size, max_len), &self.device)?;
        let token_type_ids = input_ids.zeros_like()?; // 强制为全 0 张量

        // eprintln!(
        //     "[DEBUG] 张量形状 - input_ids: {:?}, token_type_ids: {:?}",
        //     input_ids.shape(),
        //     token_type_ids.shape()
        // );

        // 5. 模型前向传播与推理输出检查
        let output = self.model.forward(&input_ids, &token_type_ids, None)?;
        // eprintln!("[DEBUG] Model Output 形状: {:?}", output.shape());

        let cls_emb = output.i((.., 0))?.contiguous()?;
        // eprintln!("[DEBUG] CLS Embedding 形状: {:?}", cls_emb.shape());

        let hidden = self.dense.forward(&cls_emb)?.tanh()?;
        let logits = self.out_proj.forward(&hidden)?;
        // eprintln!("[DEBUG] 分类 Logits 形状: {:?}", logits.shape());

        let scores = logits.squeeze(1)?.to_vec1::<f32>()?;
        Ok(scores)
    }
}
