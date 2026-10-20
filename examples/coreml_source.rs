//! Load an offline WebNN bundle with the same source compiler used by native apps.
#[cfg(all(
    target_vendor = "apple",
    feature = "coreml-runtime",
    feature = "coreml-export"
))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustnn::executors::coreml::{CoremlComputeUnits, CoremlNativeSession};
    let args = std::env::args().collect::<Vec<_>>();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: coreml_source graph.webnn|graph.json [cpu|gpu|ane|all]".into());
    }
    let policy = match args.get(2).map(String::as_str).unwrap_or("cpu") {
        "cpu" => CoremlComputeUnits::CpuOnly,
        "gpu" => CoremlComputeUnits::CpuAndGpu,
        "ane" => CoremlComputeUnits::CpuAndNeuralEngine,
        "all" => CoremlComputeUnits::All,
        _ => return Err("unknown compute policy".into()),
    };
    let start = std::time::Instant::now();
    let session =
        CoremlNativeSession::compile_webnn_source_with_progress(&args[1], policy, |phase| {
            eprintln!("{phase}: {:.3}s", start.elapsed().as_secs_f64())
        })?;
    println!(
        "Loaded locally in {:.3}s; {:?}. This load check does not qualify prediction accuracy or placement.",
        start.elapsed().as_secs_f64(),
        session.load_diagnostics()
    );
    Ok(())
}
#[cfg(not(all(
    target_vendor = "apple",
    feature = "coreml-runtime",
    feature = "coreml-export"
)))]
fn main() {
    eprintln!("Requires an Apple target and coreml-runtime,coreml-export features.");
}
