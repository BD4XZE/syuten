use crate::ResourceError;
use crate::{CandidateScore, Reranker};
use anyhow::{Context, Result};

use candle_core::Device;
use regex::Regex;
use serde::Serialize;
use std::collections::HashMap;

type Dictionary = HashMap<String, Vec<String>>;
const ONNX_DICT: &str = include_str!("data/onnx_dict_extract.json");

pub fn candle_run(word: String) -> Result<Vec<CandidateScore>> {
    // TODO:增加cli参数，更改词典为有英文解析，例句的版本.
    let candle_str = CandleStrTool::new(word.clone());

    let meanings = candle_str
        .find_word_chinese_meaning()
        .map_err(|_e| anyhow::anyhow!("word '{}' not find", &word))?;
    let sentences: Vec<String> = candle_str.get_sentences()?;
    let device = Device::new_cuda(0).context("failed to initialize CUDA")?;
    // TODO：路径硬编码，应该改正
    let model_dir = "/home/orionchen/RustroverProjects/syuten/src/model";
    let reranker = Reranker::new(model_dir, device)?;

    //创建一个数组记得加mut
    let mut result = Vec::new();


    for sentence in sentences.clone() {
        //这里修改了模型的注意力，先把目标单词放到开头，有助于提升准确度。
        let sentence = format!("<{}> | {}",&word,&sentence);
        let scores = reranker.compute_scores(&sentence, &meanings)?;
        let res = crate::get_max(&meanings, &scores, sentence)?;
        result.push(res);
    }
    Ok(result)
}

pub fn init_candle() -> Result<Reranker> {
    let device = candle_core::Device::new_cuda(0).context("failed to initialize CUDA")?;
    let model_dir = "/home/orionchen/RustroverProjects/syuten/src/model";
    let reranker = Reranker::new(model_dir, device).context("config model error")?;
    Ok(reranker)
}

pub struct CandleStrTool {
    pub word: String,
}

impl CandleStrTool {
    pub fn new(word: String) -> Self {
        CandleStrTool { word }
    }

    /// 获取目标单词所有可能的形式
    /// TODO: 未来如果有能力做到根据语法来获取单词的同一种意思的不同形式
    /// TODO：如果你要查询abandon单词，就看词典，abandoned如果是新的意思，就不查找，如果不是，就查找。
    /// TODO：每个单词都有专有的regex
    fn build_word_regex(&self) -> Result<regex::Regex, regex::Error> {
        let word = self.word.to_lowercase();
        let pattern = format!(r"(?i)\b(?:{})\b", word);
        Regex::new(&pattern)
    }

    ///输出默认词典文件,柯林斯词典.
    fn load_dict() -> Result<Dictionary, Box<dyn std::error::Error>> {
        let dict: Dictionary = serde_json::from_str(ONNX_DICT)?;
        Ok(dict)
    }

    // TODO:增加debug模式输出详细信息
    pub fn find_word_chinese_meaning(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        // 1. 先拿到字典（失败就直接抛错退出）
        let dict = Self::load_dict()?;

        // 2. 去字典里查找这个词
        match dict.get(&self.word) {
            Some(meanings) => Ok(meanings.clone()),
            None => Err(ResourceError::DictNotFound.into()),
        }
    }
    /// 使用近25年的考研英语数据集
    /// 读取 CSV 文件并提取第二列数据
    /// AI generated this function
    /// csv文件第一列应该是原始真题句子，第二列是词形还原的句子
    pub fn get_sentences(&self) -> Result<Vec<String>> {
        let mut rdr = csv::Reader::from_reader(crate::PEGG_SENTENCES.as_bytes());
        let re = self.build_word_regex()?;
        let mut filtered_sentences = Vec::new();

        for result in rdr.records() {
            let record = result?;

            // TODO:按照列名读取，而不是索引
            //这里的数据有两列，第一列是原始句子，第二列是已经词形还原处理好的句子
            //词形还原指的是根据句子的意思，把里面的单词还原成原始状态，具体参考NLP
            //领域的spacy的python库
            if let Some(restor_sentence) = record.get(1) {
                if re.is_match(&restor_sentence.to_lowercase()) {
                    if let Some(sentence) = record.get(0) {
                        filtered_sentences.push(sentence.to_string());
                    }
                }
            }
        }

        if filtered_sentences.is_empty() {
            Err(ResourceError::SentencesNotFound.into())
        } else {
            Ok(filtered_sentences)
        }
    }
}

#[derive(Serialize)]
pub struct OnnxWrite {
    pub meaning: String,
    pub index: usize,
    pub text: String,
    pub score: f32,
}

impl OnnxWrite {
    ///创建保存分析结果的结构体。
    pub fn new(candidate: CandidateScore, word: &str) -> Self {
        let CandidateScore {
            index,
            meaning: text,
            score,
            sentence,
        } = candidate;
        OnnxWrite {
            meaning: word.to_string(),
            index,
            text,
            score,
        }
    }

    ///这个函数用于批量化分析单词,结果会保存在csv里.
    pub fn write_csv(words: Vec<&str>, save_path: &str) -> Result<()> {
        let mut wtr = csv::Writer::from_path(save_path)?;

        //candle_init
        let device = Device::new_cuda(0).context("failed to initialize CUDA")?;
        let model_dir = "/home/orionchen/RustroverProjects/syuten/src/model";
        let reranker = Reranker::new(model_dir, device)?;
        for word in words {
            let candle_str = CandleStrTool::new(word.to_string());

            let candle = Self::candle_init(word, &reranker, candle_str)?;
            let aim_text: Vec<OnnxWrite> = candle
                .into_iter()
                .map(|c| OnnxWrite::new(c, word))
                .collect();
            for v in aim_text {
                wtr.serialize(v)?
            }
            println!("已完成单词:'{}' 双语消歧", word);
        }
        wtr.flush()?;

        Ok(())
    }

    /// 解决批量化运行需要重复加载模型的问题。
    pub(crate) fn candle_init(
        word: &str,
        reranker: &Reranker,
        candle_str: CandleStrTool,
    ) -> Result<Vec<CandidateScore>> {
        let meanings = candle_str
            .find_word_chinese_meaning()
            .map_err(|_e| anyhow::anyhow!("word '{}' not fond", &word))?;
        let sentences: Vec<String> = candle_str
            .get_sentences()
            .map_err(|_e| anyhow::anyhow!("{}:sentences not found", &word))?;
        //创建一个数组记得加mut
        let mut result = Vec::new();

        for sentence in sentences.clone() {
            let scores = reranker.compute_scores(&sentence, &meanings)?;
            let res = crate::get_max(&meanings, &scores, sentence)?;
            result.push(res);
        }
        Ok(result)
    }
}
