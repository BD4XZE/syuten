use crate::Reranker;
use crate::candle_model::CandleStrTool;
use crate::candle_model::OnnxWrite;
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;
use kdam::{BarExt, tqdm};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::fs::OpenOptions;

#[pyfunction]
fn write_csv(words: Vec<String>, save_path: &str) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(save_path)
        .context("open csv file failed")?;

    // 2. 通过 WriterBuilder 创建 wtr，并禁用默认表头（防止追加时多出表头）
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);
    // candle_init
    let device = Device::new_cuda(0).context("failed to initialize CUDA")?;
    let model_dir = "/home/orionchen/RustroverProjects/syuten/src/model";
    let reranker = Reranker::new(model_dir, device).context("config model error")?;

    //进度条
    let mut pb = tqdm!(total = words.len());

    for word in words {
        // 避免多余的 to_string，直接 clone 传递所有权
        let candle_str = CandleStrTool::new(word.clone());

        let candle = OnnxWrite::candle_init(&word, &reranker, candle_str).context(format!(
            "failed to init candle,word:'{}' may be not in the dict or sentences dataset",
            word
        ))?;

        let aim_text: Vec<OnnxWrite> = candle
            .into_iter()
            .map(|c| OnnxWrite::new(c, &word))
            .collect();

        for v in aim_text {
            wtr.serialize(v).context("write csv failed")?;
        }
        pb.set_description(format!("process"));
        pb.update(1)?;
        // println!("已完成单词:'{}' 双语消歧", word);
    }

    // 可以放心地直接使用 ? 和 context
    wtr.flush().context("flush csv failed")?;

    Ok(())
}

#[pymodule]
fn libsyuten(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 这里的 wrap 也能完美识别 anyhow::Result
    m.add_function(wrap_pyfunction!(write_csv, m)?)?;
    Ok(())
}
