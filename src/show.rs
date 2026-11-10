use crate::cli::{EgSentence, Freq, Index, Meaning};
use tabled::builder::Builder;
use tabled::settings::Padding;
use tabled::settings::peaker::Priority;
use tabled::settings::{Panel, Style, Width, themes::BorderCorrection};
use terminal_size::{Height, terminal_size};
///这里牵扯到了设计问题,最终结果应该怎么展示
///我的想法是先简洁,容易获取信息,再考虑美观
///我准备还是cli加tabled展示

fn get_terminal_size() -> u16 {
    let size = terminal_size();
    if let Some((terminal_size::Width(w), Height(_))) = size {
        w
    } else {
        eprintln!("Can't get terminal width, use default width,60");
        60
    }
}

// TODO: 例句高亮目标词汇
pub fn table_display<'a>(data: &'a Vec<(Index,Meaning,EgSentence,Freq)>, word: &'a str,print_eg_sentence:bool) {
    let mut builder = Builder::default();
    if print_eg_sentence {
        builder.push_record(["Freq","Meaning","Egsentence"]);
    } else {
        builder.push_record(["Freq","Meaning"])
    }
    
    for (_, meaning, eg_sentence, freq) in data {
        if print_eg_sentence {
        builder.push_record([freq.to_string(),meaning.clone(),eg_sentence.clone()]);
    } else {
        builder.push_record([freq.to_string(),meaning.clone()])
    }
    }

    let mut table = builder.build();
    let width = get_terminal_size();
    // 保留标题与边框样式
    table.with(Panel::header(word));
    table.with(BorderCorrection::span());
    table.with(Style::modern());

    // 去除单元格内边距，使整表宽度约束更精确
    table.with(Padding::zero());

    // 让整张表适应当前终端宽度
    let width = get_terminal_size() as usize;

    table.with((
        Width::wrap(width)
            .priority(Priority::max(true))
            .keep_words(true),
        Width::increase(width),
    ));
    println!("{table}");
}
