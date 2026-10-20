//! Compile a native feature-handle routing plan. Slots distinguish successive
//! definitions of the same feature name; releasing a consumed input must never
//! release a newly produced value with that name.
use crate::protos::coreml::specification::{Model, model};
use prost::Message;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

// The native loader invokes this synchronously on its calling thread. No
// observer survives compilation, and tensor contents are never passed here.
pub(super) struct LoadObserver<'a> {
    callback: Option<&'a mut dyn FnMut(&str)>,
}
impl<'a> LoadObserver<'a> {
    pub(super) fn new(callback: Option<&'a mut dyn FnMut(&str)>) -> Self {
        Self { callback }
    }
    pub(super) fn callback(
        &self,
    ) -> Option<extern "C" fn(*mut std::ffi::c_void, usize, usize, bool) -> bool> {
        self.callback.as_ref().map(|_| load_progress as _)
    }
    pub(super) fn context(&mut self) -> *mut std::ffi::c_void {
        (self as *mut Self).cast()
    }
}
extern "C" fn load_progress(
    context: *mut std::ffi::c_void,
    index: usize,
    total: usize,
    done: bool,
) -> bool {
    if context.is_null() {
        return false;
    }
    // SAFETY: the synchronous loader receives a pointer to the live observer.
    let observer = unsafe { &mut *context.cast::<LoadObserver<'_>>() };
    let Some(callback) = observer.callback.as_mut() else {
        return true;
    };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        callback(&format!(
            "child/{index}/{total}/{}",
            if done { "done" } else { "start" }
        ));
    }))
    .is_ok()
}

#[derive(Debug, Serialize)]
pub(super) struct Binding {
    pub name: String,
    pub slot: usize,
}
#[derive(Debug, Serialize)]
pub(super) struct Child {
    pub path: String,
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub release: Vec<usize>,
}
#[derive(Debug, Serialize)]
pub(super) struct Plan {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub children: Vec<Child>,
    pub slots: usize,
}

pub(super) fn from_bytes(bytes: &[u8]) -> Result<Option<Plan>, String> {
    let model = Model::decode(bytes).map_err(|e| format!("invalid CoreML protobuf: {e}"))?;
    from_model(&model)
}

fn from_model(model: &Model) -> Result<Option<Plan>, String> {
    let Some(model::Type::Pipeline(pipeline)) = &model.r#type else {
        return Ok(None);
    };
    if pipeline.models.is_empty() {
        return Err("empty native pipeline".into());
    }
    let desc = model
        .description
        .as_ref()
        .ok_or("pipeline has no description")?;
    let mut slots = Vec::new();
    let mut visible = BTreeMap::new();
    let mut inputs = Vec::new();
    for input in &desc.input {
        if input.name.is_empty() || input.r#type.is_none() || visible.contains_key(&input.name) {
            return Err("pipeline input is missing, untyped or duplicated".into());
        }
        let slot = slots.len();
        slots.push(input.r#type.clone());
        visible.insert(input.name.clone(), slot);
        inputs.push(Binding {
            name: input.name.clone(),
            slot,
        });
    }
    let mut children = Vec::new();
    let mut last_use = BTreeMap::new();
    for (index, child) in pipeline.models.iter().enumerate() {
        if !matches!(&child.r#type, Some(model::Type::MlProgram(_))) {
            return Err("native forwarding currently requires flat MLProgram children".into());
        }
        let desc = child
            .description
            .as_ref()
            .ok_or("child has no description")?;
        let mut child_inputs = Vec::new();
        let mut seen = BTreeSet::new();
        for input in &desc.input {
            if !seen.insert(&input.name) {
                return Err("duplicate child input".into());
            }
            let slot = *visible
                .get(&input.name)
                .ok_or_else(|| format!("unavailable child input {}", input.name))?;
            if slots[slot] != input.r#type {
                return Err(format!("child input type differs: {}", input.name));
            }
            last_use.insert(slot, index);
            child_inputs.push(Binding {
                name: input.name.clone(),
                slot,
            });
        }
        let mut child_outputs = Vec::new();
        let mut seen = BTreeSet::new();
        for output in &desc.output {
            if output.name.is_empty() || output.r#type.is_none() || !seen.insert(&output.name) {
                return Err("child output is missing, untyped or duplicated".into());
            }
            let slot = slots.len();
            slots.push(output.r#type.clone());
            visible.insert(output.name.clone(), slot);
            last_use.insert(slot, index); // dead outputs are released immediately
            child_outputs.push(Binding {
                name: output.name.clone(),
                slot,
            });
        }
        children.push(Child {
            path: format!("model{index}"),
            inputs: child_inputs,
            outputs: child_outputs,
            release: Vec::new(),
        });
    }
    let mut outputs = Vec::new();
    let mut seen = BTreeSet::new();
    for output in &desc.output {
        if !seen.insert(&output.name) {
            return Err("duplicate pipeline output".into());
        }
        let slot = *visible
            .get(&output.name)
            .ok_or_else(|| format!("unavailable pipeline output {}", output.name))?;
        if slots[slot] != output.r#type {
            return Err(format!("pipeline output type differs: {}", output.name));
        }
        last_use.remove(&slot); // keep until the returned provider owns it
        outputs.push(Binding {
            name: output.name.clone(),
            slot,
        });
    }
    for (slot, index) in last_use {
        children[index].release.push(slot);
    }
    Ok(Some(Plan {
        inputs,
        outputs,
        children,
        slots: slots.len(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::coreml::specification::{
        FeatureDescription, FeatureType, ModelDescription, Pipeline,
    };
    #[test]
    fn load_progress_is_synchronous_and_contains_callback_panics() {
        let mut rows = Vec::new();
        {
            let mut callback = |s: &str| rows.push(s.to_string());
            let mut observer = LoadObserver::new(Some(&mut callback));
            let invoke = observer.callback().unwrap();
            assert!(invoke(observer.context(), 2, 5, false));
            assert!(invoke(observer.context(), 2, 5, true));
        }
        assert_eq!(rows, ["child/2/5/start", "child/2/5/done"]);
        assert!(LoadObserver::new(None).callback().is_none());
        assert!(!load_progress(std::ptr::null_mut(), 0, 1, false));
        let mut failing = |_: &str| panic!("test setup observer");
        let mut observer = LoadObserver::new(Some(&mut failing));
        assert!(!load_progress(observer.context(), 0, 1, false));
    }

    fn desc(inputs: &[&str], outputs: &[&str]) -> ModelDescription {
        let feature = |name: &&str| FeatureDescription {
            name: (*name).into(),
            r#type: Some(FeatureType::default()),
            ..Default::default()
        };
        ModelDescription {
            input: inputs.iter().map(feature).collect(),
            output: outputs.iter().map(feature).collect(),
            ..Default::default()
        }
    }
    fn child(inputs: &[&str], outputs: &[&str]) -> Model {
        Model {
            description: Some(desc(inputs, outputs)),
            r#type: Some(model::Type::MlProgram(Default::default())),
            ..Default::default()
        }
    }
    fn pipeline(children: Vec<Model>) -> Model {
        Model {
            description: Some(desc(&["x"], &["x", "z"])),
            r#type: Some(model::Type::Pipeline(Pipeline {
                models: children,
                ..Default::default()
            })),
            ..Default::default()
        }
    }
    #[test]
    fn same_name_outputs_and_fanout_keep_the_correct_generation() {
        let model = pipeline(vec![
            child(&["x"], &["a", "dead"]),
            child(&["x", "a"], &["x"]),
            child(&["a", "x"], &["z"]),
        ]);
        let p = from_model(&model).unwrap().unwrap();
        assert_eq!(p.children[0].release, [2]);
        assert_eq!(p.children[1].release, [0]);
        assert_eq!(p.children[2].release, [1]);
        assert_eq!(p.outputs.iter().map(|b| b.slot).collect::<Vec<_>>(), [3, 4]);
        assert_eq!(
            p.children[2]
                .inputs
                .iter()
                .map(|b| b.slot)
                .collect::<Vec<_>>(),
            [1, 3]
        );
    }
    #[test]
    fn rejects_unavailable_features_and_mismatched_types() {
        assert!(
            from_model(&pipeline(vec![child(&["later"], &["z"])]))
                .unwrap_err()
                .contains("unavailable")
        );
        let mut model = pipeline(vec![child(&["x"], &["z"])]);
        model.description.as_mut().unwrap().output[0].r#type = None;
        assert!(from_model(&model).unwrap_err().contains("type differs"));
    }
    #[test]
    fn rejects_duplicates_empty_and_nested_without_silently_falling_back() {
        assert!(from_model(&pipeline(vec![])).is_err());
        assert!(from_model(&pipeline(vec![child(&["x", "x"], &["z"])])).is_err());
        assert!(from_model(&pipeline(vec![child(&["x"], &["z", "z"])])).is_err());
        assert!(from_model(&pipeline(vec![pipeline(vec![])])).is_err());
        assert!(from_bytes(b"invalid protobuf").is_err());
        assert!(from_model(&child(&["x"], &["z"])).unwrap().is_none());
    }
}

#[cfg(all(test, target_vendor = "apple"))]
mod native_tests {
    use super::super::{CoremlByteInput, CoremlComputeUnits, run_coreml_bytes};
    use super::*;
    use crate::converters::{CoremlMlProgramConverter, GraphConverter};
    use crate::mlcontext::{MLGraphBuilder, MLNamedOperands, MLOperandDescriptor};
    use crate::operator_enums::MLOperandDataType;
    use crate::protos::coreml::specification::Pipeline;
    use std::collections::HashMap;
    fn child(input: &str, output: &str) -> Model {
        let mut b = MLGraphBuilder::new_uncompiled();
        let desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
        let x = b.input(input, &desc).unwrap();
        let y = b.add(x, x).unwrap();
        let g = b
            .finish_graph_info(&MLNamedOperands::from([(output, y)]))
            .unwrap();
        let c = CoremlMlProgramConverter.convert(&g).unwrap();
        assert!(c.weights_data.is_none());
        Model::decode(c.data.as_slice()).unwrap()
    }
    #[test]
    fn failed_child_load_preserves_error_across_autorelease_pool() {
        // The first full-model load failed validation, then crashed while
        // copying its already-freed NSError. A missing child exercises that
        // same native error path without relying on compiler diagnostics.
        let directory = tempfile::tempdir().unwrap();
        let plan = br#"{"slots":0,"inputs":[],"outputs":[],"children":[{"path":"missing.mlmodelc","inputs":[],"outputs":[],"release":[]}]}"#;
        for _ in 0..8 {
            objc::rc::autoreleasepool(|| unsafe {
                use objc::runtime::Object;
                use objc::{class, msg_send, sel, sel_impl};
                let url = super::super::nsurl_from_path(directory.path()).unwrap();
                let config: *mut Object = msg_send![class!(MLModelConfiguration), new];
                let _guard = super::super::ReleaseOnDrop(config);
                let mut output = std::ptr::null_mut();
                let mut error = [0u8; 1024];
                let status = super::super::rustnn_coreml_load_pipeline(
                    url,
                    config,
                    plan.as_ptr(),
                    plan.len(),
                    None,
                    std::ptr::null_mut(),
                    &mut output,
                    error.as_mut_ptr().cast(),
                    error.len(),
                );
                assert_ne!(status, 0);
                assert!(output.is_null());
                assert!(!super::super::shim_error_to_string(&error).is_empty());
            });
        }
    }
    #[test]
    fn forwards_native_features_and_retains_outputs_across_child_pools() {
        let first = child("x", "hidden");
        let last = child("hidden", "y");
        let mut description = first.description.clone().unwrap();
        description.output = last.description.clone().unwrap().output;
        // Keep an early child output too: it must survive its last consumer.
        description
            .output
            .extend(first.description.clone().unwrap().output);
        let model = Model {
            specification_version: 9,
            description: Some(description),
            r#type: Some(model::Type::Pipeline(Pipeline {
                models: vec![first, last],
                ..Default::default()
            })),
            ..Default::default()
        };
        let mut progress = Vec::new();
        let mut callback = |phase: &str| progress.push(phase.to_owned());
        let mut trace = super::super::load::LoadTrace::strict(CoremlComputeUnits::CpuOnly);
        let native = super::super::compile_model_from_url_observed(
            model.encode_to_vec(),
            None::<Vec<u8>>,
            &mut trace,
            Some(&mut callback),
        )
        .unwrap();
        assert_eq!(
            progress,
            [
                "child/0/2/start",
                "child/0/2/done",
                "child/1/2/start",
                "child/1/2/done"
            ]
        );
        let desc = crate::graph::OperandDescriptor {
            data_type: crate::graph::DataType::Float32,
            shape: vec![crate::graph::Dimension::Static(4)],
            pending_permutation: vec![],
        };
        for step in 0..32 {
            let input = [step as f32; 4]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let outputs =
                HashMap::from([("y".into(), desc.clone()), ("hidden".into(), desc.clone())]);
            let actual = run_coreml_bytes(
                &native,
                &HashMap::from([(
                    "x".into(),
                    CoremlByteInput {
                        data: &input,
                        descriptor: &desc,
                    },
                )]),
                &outputs,
            )
            .unwrap();
            for (name, delta) in [("hidden", 2.0), ("y", 4.0)] {
                for b in actual[name].chunks_exact(4) {
                    assert_eq!(
                        f32::from_le_bytes(b.try_into().unwrap()),
                        step as f32 * delta
                    );
                }
            }
        }
    }
}
