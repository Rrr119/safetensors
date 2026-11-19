use criterion::{criterion_group, criterion_main, Criterion};
use safetensors::incremental::IncrementalSafeTensors;
use safetensors::tensor::{serialize_to_file, Dtype, SafeTensors, TensorView};
use std::collections::HashMap;
use std::hint::black_box;
use tempfile::NamedTempFile;

const TENSOR_BYTES: usize = 2 * 1024 * 1024;
const TENSOR_COUNT: usize = 16;

fn make_file() -> (NamedTempFile, HashMap<String, TensorView<'static>>) {
    // Leaked on purpose: the benchmark process is short-lived and the views
    // must outlive the setup call.
    let data: &'static [u8] = Box::leak(vec![7u8; TENSOR_BYTES].into_boxed_slice());
    let mut tensors = HashMap::new();
    for index in 0..TENSOR_COUNT {
        tensors.insert(
            format!("model.layers.{index}.weight"),
            TensorView::new(Dtype::U8, vec![TENSOR_BYTES], data).unwrap(),
        );
    }

    let file = NamedTempFile::new().unwrap();
    serialize_to_file(tensors.iter(), None, file.path()).unwrap();
    (file, tensors)
}

fn bench_incremental(c: &mut Criterion) {
    let (file, tensors) = make_file();
    let bytes = std::fs::read(file.path()).unwrap();
    let name = "model.layers.7.weight";

    let mut group = c.benchmark_group("incremental-load");

    group.bench_function("full deserialize 32_MB", |b| {
        b.iter(|| SafeTensors::deserialize(black_box(&bytes)).unwrap());
    });

    group.bench_function("header only 32_MB", |b| {
        b.iter(|| SafeTensors::read_metadata_from_file(file.path()).unwrap());
    });

    group.bench_function("one tensor, pread 32_MB", |b| {
        let loader = IncrementalSafeTensors::open_pread(file.path()).unwrap();
        b.iter(|| loader.read_tensor(black_box(name)).unwrap());
    });

    group.bench_function("open + one tensor, pread 32_MB", |b| {
        b.iter(|| {
            let loader = IncrementalSafeTensors::open_pread(file.path()).unwrap();
            loader.read_tensor(black_box(name)).unwrap()
        });
    });

    #[cfg(feature = "mmap")]
    group.bench_function("one tensor, mmap 32_MB", |b| {
        let loader = IncrementalSafeTensors::open_mmap(file.path()).unwrap();
        b.iter(|| loader.read_tensor(black_box(name)).unwrap());
    });

    group.bench_function("all tensors, pread incremental", |b| {
        let loader = IncrementalSafeTensors::open_pread(file.path()).unwrap();
        b.iter(|| loader.tensors().unwrap().len());
    });

    group.bench_function("serialize 32_MB", |b| {
        b.iter(|| {
            let out = file.path().with_extension("bench.safetensors");
            serialize_to_file(black_box(tensors.iter()), None, &out).unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_incremental);
criterion_main!(benches);
