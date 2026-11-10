use crate::{
    CandidateScore, Reranker,
    candle_model::{CandleStrTool, init_candle},
    cli,
};
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::seq::{IndexedRandom, SliceRandom};
use std::{
    collections::HashMap,
    io::{self, Write},
};
type ResUnit = (Index,Meaning,EgSentence,Freq);
pub(crate) type UnitData = (Index, Meaning, EgSentence, Score);
pub(crate) type Meaning = String; // 单词的汉语意思
pub(crate) type EgSentence = String; // 单词的随机真题例句（目前是随机的，因为显示空间有限）
type Score = i32;
pub(crate) type Freq = usize; 
pub(crate) type Index = usize;
/// syuten，a project serve for PEGG
#[derive(Debug, Parser)]
struct Args {
    word: Option<String>,

    #[arg(short, long, default_value_t = false)]
    detail: bool,

    #[arg(short, long, default_value_t = false)]
    sentences: bool,
}

//这里到时候用newtype
fn get_random_sentence(
    data: ((Index, String), Freq),
    cand: Vec<CandidateScore>,
) -> Result<EgSentence> {
    let meaning = data.0.1;
    let mut matches = rand::rng();
    // 这个sentence hashmap第一个数据是meaning,第二个是sentence,真题例句
    let data: Vec<(Meaning, EgSentence)> =
        cand.into_iter().map(|x| (x.meaning, x.sentence)).collect();
    let meanings: Vec<EgSentence> = data
        .into_iter()
        .filter(|(m, _)| *m == meaning)
        .map(|(_, v)| v)
        .collect();
    let res = meanings
        .choose(&mut matches)
        .context("random chose error")?;
    Ok(res.clone())
}
///处理candle输出的原始数据，进行统计展示频率最高的前5项
/// 这里的print sentences是只输出出现词频<=2的中文意思。且为了避免屏幕空间不足
/// 一次只随机输出一个例句，未来可以添加参数输出全部例句
pub fn result_process(
    cand: Vec<CandidateScore>,
) -> Result<Vec<(Index,Meaning,EgSentence,Freq)>> {
    //这里就不需要Score了，因为Score的作用就是排序，后续不应该出现Score
    let res: HashMap<(Index, Meaning), Freq> = cand.iter().fold(
        HashMap::new(),
        |mut acc: HashMap<(Index, Meaning), Freq>, cand| {
            let freq = (cand.index, cand.meaning.clone());
            *acc.entry(freq).or_insert(0) += 1;
            acc
        },
    );
    let counts: Result<Vec<(Index,Meaning,EgSentence,Freq)>> = res
        .into_iter()
        .map(|x:((Index, Meaning), Freq)| {
            Ok((
                x.0.0,
                x.0.1.clone(),
                get_random_sentence(x.clone(), cand.clone())?,
                x.1,
            ))
        })
        .collect(); 
    let mut counts: Vec<(Index,Meaning,EgSentence,Freq)> = counts?;
    counts.sort_by(|a: &(Index,Meaning,EgSentence,Freq) ,b: &(Index,Meaning,EgSentence,Freq)| b.3.cmp(&a.3));

    //获取频率最高的前5项
    if counts.len() <= 5 {
        return Ok(counts);
    } else {
        return Ok(counts.drain(0..=4).collect());
    }
}

/// 主要的cli操作入口
/// 主要的性能占用是模型启动时间，占用了大概96%的时间
pub fn cli_run() -> Result<()> {
    let args = Args::parse();

    if let Some(word) = args.word {
        let word_strip = word.trim();
        let onnx = crate::candle_model::candle_run(word_strip.to_string())?;
        let res: Vec<_> = cli::result_process(onnx)?;
        crate::show::table_display(&res, word_strip,args.sentences);
    } else {
        println!("Entered interactive mode. Ctrl+D or Ctrl+C to exit\n");
        let stdin = io::stdin();
        let reranker = init_candle()?;
        let _ = cli_loop(stdin, reranker,args.sentences);
    }
    Ok(())
}

///实现了cli_run函数的loop功能
fn cli_loop(stdin: io::Stdin, reranker: Reranker,pr_eg_sentence:bool) -> Result<()> {
    loop {
        print!("> ");
        io::stdout().flush()?; // 确保提示符立即输出

        let mut input = String::new();
        if stdin.read_line(&mut input)? == 0 {
            break; // EOF (Ctrl+D)
        }

        let word = input.trim();
        if word.is_empty() {
            continue;
        }

        let candle_str = CandleStrTool::new(word.to_string());
        let meanings = match candle_str.find_word_chinese_meaning() {
            Ok(meaning) => meaning,
            Err(e) => {
                eprintln!("word {} not found: {}", word, e);
                continue;
            }
        };
        let sentences: Vec<String> = match candle_str.get_sentences() {
            Ok(meaning) => meaning,
            Err(e) => {
                eprintln!("word {} not found: {}", word, e);
                continue;
            }
        };

        let mut results = Vec::new();
        for sentence in sentences.clone() {
            let scores = reranker.compute_scores(&sentence, &meanings)?;
            let res = crate::get_max(&meanings, &scores, sentence)?;
            results.push(res);
        }
        let res: Vec<ResUnit> = cli::result_process(results)?;

        crate::show::table_display(&res, word,pr_eg_sentence);
    }

    Ok(())
}
pub fn manual_run(word: &str) -> Result<()> {
    let onnx = crate::candle_model::candle_run(word.to_string())?;
    let res: Vec<ResUnit> = cli::result_process(onnx)?;
    crate::show::table_display(&res, word,false);
    Ok(())
}
