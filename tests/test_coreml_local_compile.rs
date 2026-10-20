//! Exercise source parsing, local bundle emission and native carried tensors.
#![cfg(all(
    target_vendor = "apple",
    feature = "coreml-runtime",
    feature = "coreml-export"
))]

use rustnn::executors::coreml::CoremlLoadRoute;
use rustnn::mlcontext::{
    Backend, BackendStatistics, LoadDiagnostics, MLContext, MLContextOptions, MLGraphBuilder,
    MLNamedOperands, MLNamedTensors, MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
    RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;

#[test]
fn source_bundle_compiles_locally_and_carries_native_state_for_six_turns() {
    let mut options = RustNNOptions::default();
    options.coreml.reuse_tensor_storage = true;
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml)
            .with_rustnn_options(options),
    )
    .unwrap();
    let source = tempfile::tempdir().unwrap();
    let path = source.path().join("carried.webnn");
    // The deployed input is WebNN text plus its SafeTensors sidecar. Nothing
    // below reads a pre-lowered model or invokes a host compiler subprocess.
    let increment = [0.25f32, -0.5, 1.0, -2.0];
    {
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
        let input = builder.input("carry", &desc).unwrap();
        let delta = builder.constant_from_slice(&desc, &increment).unwrap();
        let output = builder.add(input, delta).unwrap();
        builder
            .rustnn_save_webnn(&MLNamedOperands::from([("next", output)]), &path)
            .unwrap();
    }
    let info = rustnn::load_graph_from_path(&path).unwrap();
    assert!(path.with_extension("safetensors").is_file());
    let mut graph = context.rustnn_build_graph(info).unwrap();
    let Some(LoadDiagnostics::Coreml(load)) = graph.rustnn_load_diagnostics() else {
        panic!("missing diagnostics")
    };
    assert_eq!(load.route, CoremlLoadRoute::RustExport);
    assert_eq!(load.loaded_compute_units, "CPU_ONLY");
    let desc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
        .to_writable()
        .to_readable();
    let mut state = context.create_tensor(&desc).unwrap();
    let mut next = context.create_tensor(&desc).unwrap();
    context.write_tensor(&state, &[0f32; 4]).unwrap();
    for turn in 1..=6 {
        for _ in 0..32 {
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("carry", &state)]),
                    &MLNamedTensors::from([("next", &next)]),
                )
                .unwrap();
            std::mem::swap(&mut state, &mut next);
        }
        let mut actual = [0f32; 4];
        context.read_tensor(&state, &mut actual).unwrap();
        // Exact dyadic reference. Only the final result of each turn is read;
        // no read/write is used to carry state between predictions.
        assert_eq!(actual, increment.map(|x| x * (turn * 32) as f32));
        let Some(BackendStatistics::Coreml(stats)) = context.rustnn_backend_statistics() else {
            panic!("missing statistics")
        };
        assert_eq!(stats.host_write_bytes, 16);
        assert_eq!(stats.host_read_bytes, turn * 16);
        assert_eq!(stats.input_copy_bytes, 0);
        assert_eq!(stats.output_copy_bytes, 0);
        assert_eq!(stats.output_backings_accepted, turn * 32);
        assert_eq!(stats.native_input_bindings, turn * 32);
    }
}
