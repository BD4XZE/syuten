use criterion::{Criterion, criterion_group, criterion_main};
use syuten::cli::manual_run;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("manual_run_group");
    group.sample_size(10);

    group.bench_function("fib 5", |b: &mut criterion::Bencher<'_>| {
        b.iter(|| manual_run("finding"))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
