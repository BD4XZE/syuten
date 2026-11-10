use anyhow::Result;
use rayon::prelude::*;
use syuten::candle_model::CandleStrTool;
use syuten::candle_model::OnnxWrite;
use syuten::cli::{cli_run, manual_run};

/// 词频表数据编译期嵌入,避免运行时路径依赖
const PEGG_FREQUENCY: &str = include_str!("data/PEGG_frequency.csv");

/// TODO：再加上log功能，提示哪个单词没有识别出。
/// 实际上编译前就能检查出的。毕竟数据已经嵌入二进制编译文件中了。
fn filter_words(words: Vec<&str>) -> Vec<&str> {
    words
        .into_par_iter() // 1. 开启多线程并行迭代
        .filter(|&word| {
            let candle_str = CandleStrTool::new(word.to_string());
            candle_str.find_word_chinese_meaning().is_ok() && candle_str.get_sentences().is_ok()
        })
        .collect()
}
fn csv_analysis() -> Result<()> {
    let save_path = "./cash_tqdm.csv";
    let words = PEGG_FREQUENCY
        .lines()
        .skip(1) // 跳过第一行表头
        .filter_map(|line| line.split(',').next()) // 取逗号分隔的第一列
        .collect();
    let clean_words = filter_words(words);
    OnnxWrite::write_csv(clean_words, save_path)
}

fn main() -> Result<()> {
    cli_run()
}
