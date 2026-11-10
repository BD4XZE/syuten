use anyhow::{Ok, Result};
use syuten::CandidateScore;

#[test]
fn test_sort() -> Result<()> {
    let meanings = vec![
        "1 [N-UNCOUNT 不可数名词] 自由;自由行为；".to_string(),
        "2 [N-UNCOUNT 不可数名词] 人身自由;自由身份；".to_string(),
        "3 [N-UNCOUNT 不可数名词] 摆脱;免除;不受...影响的状态；".to_string(),
        "4 [N-SING 单数名词] (某市的)荣誉市民称号；".to_string(),
    ];
    let sentences = "TODO".to_string();
    let scores = vec![4.0, 3.0, 2.0, 1.0];
    let res = syuten::get_max(&meanings, &scores, sentences)?;
    assert_eq!(
        res,
        syuten::CandidateScore {
            index: 0,
            meaning: "1 [N-UNCOUNT 不可数名词] 自由;自由行为；".to_string(),
            score: 4.0,
            sentence: "todo".to_string(),
        }
    );
    Ok(())
}

// #[test]
// ///这个test写的我恶心
// fn test_manual_run() -> Result<()> {
//     let onnx = vec![
//         CandidateScore {
//             index: 0,
//             meaning: "1 [N-UNCOUNT 不可数名词] 自由;自由行为；".to_string(),
//             score: 0.98363173,
//         },
//         CandidateScore {
//             index: 0,
//             meaning: "1 [N-UNCOUNT 不可数名词] 自由;自由行为；".to_string(),
//             score: 2.0140615,
//         },
//         CandidateScore {
//             index: 0,
//             meaning: "1 [N-UNCOUNT 不可数名词] 自由;自由行为；".to_string(),
//             score: 2.7222302,
//         },
//         CandidateScore {
//             index: 1,
//             meaning: "2 [N-UNCOUNT 不可数名词] 人身自由;自由身份；".to_string(),
//             score: 2.653295,
//         },
//         CandidateScore {
//             index: 0,
//             meaning: "1 [N-UNCOUNT 不可数名词] 自由;自由行为；".to_string(),
//             score: 2.2092447,
//         },
//         CandidateScore {
//             index: 0,
//             meaning: "1 [N-UNCOUNT 不可数名词] 自由;自由行为；".to_string(),
//             score: -0.91433275,
//         },
//         CandidateScore {
//             index: 3,
//             meaning: "4 [N-SING 单数名词] (某市的)荣誉市民称号；".to_string(),
//             score: 1.1435012,
//         },
//     ];
//     let res = syuten::cli::result_process(onnx);
//     assert_eq!(
//         res,
//         vec![
//             (
//                 (
//                     0 as usize,
//                     "1 [N-UNCOUNT 不可数名词] 自由;自由行为；".to_string()
//                 ),
//                 5 as i32
//             ),
//             (
//                 (
//                     3 as usize,
//                     "4 [N-SING 单数名词] (某市的)荣誉市民称号；".to_string()
//                 ),
//                 1.0 as i32
//             ),
//             (
//                 (
//                     1 as usize,
//                     "2 [N-UNCOUNT 不可数名词] 人身自由;自由身份；".to_string()
//                 ),
//                 1.0 as i32
//             ),
//         ]
//     );
//     Ok(())
// }
