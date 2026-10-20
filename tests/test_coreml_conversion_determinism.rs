//! Stable artifacts for an identical graph, not canonicalization of equivalent graphs.

use std::process::Command;

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::operators::Operation;
use rustnn::protos::coreml::{
    mil_spec::value,
    specification::{Metadata, Model, model},
};

fn graph(reverse_insertion: bool, dtype: DataType) -> GraphInfo {
    let operand = |name: &str, kind| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: dtype,
            shape: vec![Dimension::Static(2)],
            pending_permutation: vec![],
        },
    };
    let mut graph = GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input),
            operand("left", OperandKind::Constant),
            operand("right", OperandKind::Constant),
            operand("sum", OperandKind::Intermediate),
            operand("result", OperandKind::Output),
        ],
        input_operands: vec![0],
        output_operands: vec![4],
        operations: vec![
            Operation::Add {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![3],
            },
            Operation::Mul {
                a: 3,
                b: 2,
                options: None,
                outputs: vec![4],
            },
        ],
        ..Default::default()
    };
    let constants: [(u32, Vec<u8>); 2] = match dtype {
        DataType::Float16 => [
            (1, bytemuck::cast_slice(&[0x3800u16, 0x4000]).to_vec()),
            (2, bytemuck::cast_slice(&[0x4400u16, 0xbc00]).to_vec()),
        ],
        DataType::Int32 => [
            (1, bytemuck::cast_slice(&[3i32, 5]).to_vec()),
            (2, bytemuck::cast_slice(&[7i32, -1]).to_vec()),
        ],
        _ => unreachable!(),
    };
    for index in if reverse_insertion { [1, 0] } else { [0, 1] } {
        let (id, data) = &constants[index];
        graph.constant_operand_ids_to_handles.insert(
            *id,
            ConstantData {
                data: data.clone(),
                label: None,
            },
        );
    }
    graph
}

#[test]
fn conversion_worker() {
    let Ok(reverse) = std::env::var("RUSTNN_TEST_COREML_STABILITY_WORKER") else {
        return;
    };
    for dtype in [DataType::Float16, DataType::Int32] {
        let converted = CoremlMlProgramConverter
            .convert(&graph(reverse == "1", dtype))
            .unwrap();
        println!(
            "ARTIFACT:{}",
            serde_json::to_string(&(converted.data, converted.weights_data)).unwrap()
        );
    }
}

#[test]
fn metadata_worker() {
    let Ok(reverse) = std::env::var("RUSTNN_TEST_COREML_METADATA_WORKER") else {
        return;
    };
    let mut bindings = vec![
        ("rustnn.coreml.name_encoding", "hex-v1"),
        (
            "rustnn.webnn.input_aliases",
            r#"{"state":"rustnn_escaped_7374617465"}"#,
        ),
        (
            "rustnn.webnn.output_aliases",
            r#"{"reply":"physical_reply"}"#,
        ),
        (
            "rustnn.webnn.output_passthroughs",
            r#"{"reply":{"input":"state","descriptor":{"data_type":"float32","shape":[2]}}}"#,
        ),
    ];
    if reverse == "1" {
        bindings.reverse();
    }
    let converted = CoremlMlProgramConverter
        .convert(&graph(reverse == "1", DataType::Float16))
        .unwrap();
    let mut model = Model::decode(converted.data.as_slice()).unwrap();
    // Exercise the protobuf map directly: real logical-name/copy bindings can
    // add several metadata entries independently of the converter's MIL maps.
    model.description.as_mut().unwrap().metadata = Some(Metadata {
        user_defined: bindings
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect(),
        ..Default::default()
    });
    println!(
        "METADATA:{}",
        serde_json::to_string(&(model.encode_to_vec(), converted.weights_data)).unwrap()
    );
}

#[test]
fn metadata_maps_encode_identically_across_processes() {
    let mut reference = None;
    for iteration in 0..8 {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "metadata_worker", "--nocapture"])
            .env(
                "RUSTNN_TEST_COREML_METADATA_WORKER",
                (iteration % 2).to_string(),
            )
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let artifacts: Vec<(Vec<u8>, Option<Vec<u8>>)> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .filter_map(|line| line.strip_prefix("METADATA:"))
            .map(|json| serde_json::from_str(json).unwrap())
            .collect();
        assert_eq!(artifacts.len(), 1);
        assert!(artifacts[0].1.is_some(), "retain the original sidecar");
        if let Some(reference) = &reference {
            assert_eq!(&artifacts, reference, "fresh metadata process {iteration}");
        } else {
            reference = Some(artifacts);
        }
    }
}

#[test]
fn identical_graph_produces_identical_artifacts_across_processes() {
    let mut reference = None;
    for iteration in 0..8 {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "conversion_worker", "--nocapture"])
            .env(
                "RUSTNN_TEST_COREML_STABILITY_WORKER",
                (iteration % 2).to_string(),
            )
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let artifacts: Vec<(Vec<u8>, Option<Vec<u8>>)> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .filter_map(|line| line.strip_prefix("ARTIFACT:"))
            .map(|json| serde_json::from_str(json).unwrap())
            .collect();
        assert_eq!(artifacts.len(), 2);
        assert!(artifacts[0].1.is_some(), "exercise external weights");
        assert!(artifacts[1].1.is_none(), "exercise immediate constants");
        if let Some(reference) = &reference {
            assert_eq!(&artifacts, reference, "fresh process {iteration}");
        } else {
            reference = Some(artifacts);
        }
    }
}

#[test]
fn blob_references_still_address_the_exact_constant_payloads() {
    let graph = graph(true, DataType::Float16);
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let weights = converted.weights_data.unwrap();
    let model = Model::decode(converted.data.as_slice()).unwrap();
    let Some(model::Type::MlProgram(program)) = model.r#type else {
        panic!("expected MLProgram")
    };
    let block = &program.functions["main"].block_specializations["CoreML7"];
    let mut names = Vec::new();
    for operation in &block.operations {
        if operation.r#type != "const" {
            continue;
        }
        let Some(value::Value::BlobFileValue(blob)) = &operation.attributes["val"].value else {
            continue;
        };
        let name = operation.outputs[0].name.as_str();
        names.push(name);
        let id = graph
            .operands
            .iter()
            .position(|operand| operand.name.as_deref() == Some(name))
            .unwrap() as u32;
        let expected = &graph.constant_operand_ids_to_handles[&id].data;
        let metadata = usize::try_from(blob.offset).unwrap();
        assert_eq!(
            &weights[metadata..metadata + 4],
            &0xdeadbeefu32.to_le_bytes()
        );
        let length =
            u64::from_le_bytes(weights[metadata + 8..metadata + 16].try_into().unwrap()) as usize;
        let offset =
            u64::from_le_bytes(weights[metadata + 16..metadata + 24].try_into().unwrap()) as usize;
        assert_eq!(length, expected.len());
        assert_eq!(&weights[offset..offset + length], expected);
        assert_eq!(blob.file_name, "@model_path/weights/weights.bin");
    }
    assert_eq!(names, ["left", "right"]);
}
