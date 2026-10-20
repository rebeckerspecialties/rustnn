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

#[cfg(target_os = "watchos")]
#[test]
fn source_packed_products_use_bounded_native_expansion_without_carry_copies() {
    use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};

    let mut options = RustNNOptions::default();
    options.coreml.reuse_tensor_storage = true;
    // Two independent K=32 products, eight output columns per iteration.
    options.coreml.packed_weight_expansion_bytes = Some(2 * 32 * 8 * 4);
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml)
            .with_rustnn_options(options),
    )
    .unwrap();
    let source = tempfile::tempdir().unwrap();
    let path = source.path().join("packed.webnn");
    let increment: Vec<f32> = (0..64).map(|i| (i - 32) as f32 * 0.25).collect();
    {
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 1, 32]);
        let input = builder.input("carry", &desc).unwrap();
        let weights: Vec<i8> = (0..2 * 32 * 32)
            .map(|i| i8::from((i / 32) % 32 == i % 32))
            .collect();
        let weight = builder
            .constant_from_slice(
                &MLOperandDescriptor::new(MLOperandDataType::Int8, vec![2, 32, 32]),
                &weights,
            )
            .unwrap();
        let scale = builder
            .constant_from_slice(
                &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![]),
                &[1f32],
            )
            .unwrap();
        let unpacked = builder.dequantize_linear(weight, scale).unwrap();
        let product = builder.matmul(input, unpacked).unwrap();
        let delta = builder.constant_from_slice(&desc, &increment).unwrap();
        let output = builder.add(product, delta).unwrap();
        builder
            .rustnn_save_webnn(&MLNamedOperands::from([("next", output)]), &path)
            .unwrap();
    }
    let info = rustnn::load_graph_from_path(&path).unwrap();
    // Confirm admission from the real source converter, not a hand-authored MIL
    // fixture. External packed bytes must remain identical after scheduling.
    let mut converted = CoremlMlProgramConverter.convert(&info).unwrap();
    let weights = converted.weights_data.clone();
    let records = converted
        .bound_coreml_packed_matmuls(2 * 32 * 8 * 4)
        .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].tile_columns, 8);
    assert_eq!(records[0].iterations, 4);
    assert_eq!(converted.weights_data, weights);
    let mut graph = context.rustnn_build_graph(info).unwrap();
    let desc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 1, 32])
        .to_writable()
        .to_readable();
    let mut carry = context.create_tensor(&desc).unwrap();
    let mut next = context.create_tensor(&desc).unwrap();
    context.write_tensor(&carry, &[0f32; 64]).unwrap();
    for turn in 1..=6 {
        for _ in 0..32 {
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("carry", &carry)]),
                    &MLNamedTensors::from([("next", &next)]),
                )
                .unwrap();
            std::mem::swap(&mut carry, &mut next);
        }
        let mut actual = [0f32; 64];
        context.read_tensor(&carry, &mut actual).unwrap();
        assert_eq!(
            actual.as_slice(),
            increment
                .iter()
                .map(|x| x * (turn * 32) as f32)
                .collect::<Vec<_>>()
        );
        let Some(BackendStatistics::Coreml(stats)) = context.rustnn_backend_statistics() else {
            panic!("missing statistics")
        };
        assert_eq!(stats.host_write_bytes, 256);
        assert_eq!(stats.host_read_bytes, turn * 256);
        assert_eq!(stats.input_copy_bytes, 0);
        assert_eq!(stats.output_copy_bytes, 0);
        assert_eq!(stats.output_backings_accepted, turn * 32);
    }
}

#[cfg(not(target_os = "watchos"))]
#[test]
fn bounded_backend_option_rejects_non_watch_targets() {
    let mut options = RustNNOptions::default();
    options.coreml.packed_weight_expansion_bytes = Some(2048);
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml)
            .with_rustnn_options(options),
    )
    .unwrap();
    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let x = builder
        .input(
            "x",
            &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1]),
        )
        .unwrap();
    let y = builder.add(x, x).unwrap();
    let error = builder
        .build(&MLNamedOperands::from([("y", y)]))
        .unwrap_err();
    assert!(error.to_string().contains("supported only on watchOS"));
}
