//! Sidecar identity must not depend on a graph HashMap's iteration order.

use std::collections::{HashMap, HashSet};

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::operators::Operation;
use rustnn::protos::coreml::mil_spec::value;
use rustnn::protos::coreml::specification::{Model, model};

fn graph(order: &[u32]) -> GraphInfo {
    let payloads = [
        [0x0000u16, 0x8000, 0x0001, 0x8001],
        [0x3c00, 0xbc00, 0x3c01, 0xbc01],
        [0x7bff, 0xfbff, 0x7c00, 0xfc00],
        [0x3555, 0x0400, 0x8400, 0x7e01],
    ];
    let operand = |name: String, kind| Operand {
        name: Some(name),
        kind,
        descriptor: OperandDescriptor {
            data_type: DataType::Float16,
            shape: vec![Dimension::Static(4)],
            pending_permutation: vec![],
        },
    };
    let constants = order
        .iter()
        .map(|&id| {
            (
                id,
                ConstantData {
                    data: payloads[id as usize]
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                    label: None,
                },
            )
        })
        .collect();
    GraphInfo {
        operands: (0..4)
            .map(|id| operand(format!("weight_{id}"), OperandKind::Constant))
            .chain((0..4).map(|id| operand(format!("result_{id}"), OperandKind::Output)))
            .collect(),
        operations: (0..4)
            .map(|id| Operation::Identity {
                input: id,
                options: None,
                outputs: vec![id + 4],
            })
            .collect(),
        output_operands: (4..8).collect(),
        constant_operand_ids_to_handles: constants,
        ..Default::default()
    }
}

fn references(model: &[u8]) -> HashMap<String, usize> {
    let model = Model::decode(model).unwrap();
    let Some(model::Type::MlProgram(program)) = model.r#type else {
        panic!("MLProgram");
    };
    let function = &program.functions["main"];
    let block = &function.block_specializations[&function.opset];
    block
        .operations
        .iter()
        .filter_map(|operation| {
            if operation.r#type != "const" {
                return None;
            }
            let value = operation.attributes.get("val")?;
            let Some(value::Value::BlobFileValue(blob)) = &value.value else {
                return None;
            };
            assert_eq!(blob.file_name, "@model_path/weights/weights.bin");
            Some((
                operation.outputs[0].name.clone(),
                usize::try_from(blob.offset).unwrap(),
            ))
        })
        .collect()
}

fn verify_payloads(graph: &GraphInfo, model: &[u8], weights: &[u8]) {
    let references = references(model);
    assert_eq!(references.len(), 4);
    assert_eq!(u32::from_le_bytes(weights[0..4].try_into().unwrap()), 4);
    for (&id, constant) in &graph.constant_operand_ids_to_handles {
        let offset = references[&format!("weight_{id}")];
        assert_eq!(offset % 64, 0);
        assert_eq!(
            u32::from_le_bytes(weights[offset..offset + 4].try_into().unwrap()),
            0xdead_beef
        );
        assert_eq!(
            u32::from_le_bytes(weights[offset + 4..offset + 8].try_into().unwrap()),
            1,
            "stored Half dtype"
        );
        let length = usize::try_from(u64::from_le_bytes(
            weights[offset + 8..offset + 16].try_into().unwrap(),
        ))
        .unwrap();
        let payload = usize::try_from(u64::from_le_bytes(
            weights[offset + 16..offset + 24].try_into().unwrap(),
        ))
        .unwrap();
        assert_eq!(length, constant.data.len());
        assert_eq!(
            &weights[payload..payload + length],
            constant.data.as_slice()
        );
    }
}

#[test]
fn coreml_sidecar_is_identical_across_reordered_constant_maps() {
    let mut orders = HashSet::new();
    let mut expected = None;
    for index in 0..32 {
        let insertion = if index % 2 == 0 {
            [0, 1, 2, 3]
        } else {
            [3, 2, 1, 0]
        };
        let graph = graph(&insertion);
        orders.insert(
            graph
                .constant_operand_ids_to_handles
                .keys()
                .copied()
                .collect::<Vec<_>>(),
        );
        let source = serde_json::to_value(&graph).unwrap();
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let weights = converted.weights_data.unwrap();
        verify_payloads(&graph, &converted.data, &weights);
        assert_eq!(
            serde_json::to_value(&graph).unwrap(),
            source,
            "conversion does not reorder the source graph"
        );
        if let Some(expected) = &expected {
            assert_eq!(
                &weights, expected,
                "same operands and represented values must yield identical weight bytes"
            );
        } else {
            expected = Some(weights);
        }
    }
    assert!(
        orders.len() > 1,
        "the regression must exercise distinct observed HashMap orders"
    );
}

#[test]
fn coreml_sidecar_orders_original_operand_ids_and_keeps_blob_joins() {
    let graph = graph(&[3, 2, 1, 0]);
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let weights = converted.weights_data.unwrap();
    verify_payloads(&graph, &converted.data, &weights);
    let references = references(&converted.data);
    let offsets = (0..4)
        .map(|id| references[&format!("weight_{id}")])
        .collect::<Vec<_>>();
    assert!(
        offsets.windows(2).all(|pair| pair[0] < pair[1]),
        "original operand IDs define stable sidecar order"
    );
}

#[test]
fn coreml_sidecar_repeated_conversion_is_identical() {
    let graph = graph(&[2, 0, 3, 1]);
    let first = CoremlMlProgramConverter.convert(&graph).unwrap();
    for _ in 0..8 {
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        assert_eq!(converted.weights_data, first.weights_data);
        verify_payloads(
            &graph,
            &converted.data,
            converted.weights_data.as_ref().unwrap(),
        );
    }
}
