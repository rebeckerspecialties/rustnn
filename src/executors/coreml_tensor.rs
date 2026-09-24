//! Owned, reusable tensor storage for synchronous CoreML dispatch.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{HashMap, HashSet};
use std::ptr::NonNull;
use std::sync::Mutex;

use super::*;
use crate::mlcontextoptions::CoremlTensorStatistics;

fn failed(reason: impl Into<String>) -> GraphError {
    GraphError::CoremlRuntimeFailed {
        reason: reason.into(),
    }
}

fn native_code(dtype: DataType) -> Option<i32> {
    match dtype {
        DataType::Float32 => Some(65568),
        DataType::Float16 => Some(65552),
        DataType::Int32 => Some(131104),
        _ => None,
    }
}

fn same_type(dtype: DataType, code: i32) -> bool {
    matches!(
        (dtype, code),
        (DataType::Float32, 32 | 65568)
            | (DataType::Float16, 16 | 65552)
            | (DataType::Int32, 3 | 131104)
    )
}

fn is_fixed_output_shape(constraint_type: i64, enumerated_count: usize) -> bool {
    // MLMultiArrayShapeConstraintTypeEnumerated with exactly one allowed shape.
    constraint_type == 2 && enumerated_count == 1
}

/// Only this context-owned storage may mutate its array. Locks remain held
/// throughout synchronous prediction, including direct writes to output backings.
#[derive(Debug)]
pub(crate) enum CoremlTensorStorage {
    Host(Vec<u8>),
    Native(Mutex<NativeTensor>),
}

#[derive(Debug)]
pub(crate) struct NativeTensor {
    data: NonNull<u8>,
    layout: Layout,
    capacity: usize,
    dtype: DataType,
    view: *mut Object,
    shape: Vec<i64>,
}

// SAFETY: The allocation and retained view are exclusively owned. The view is
// only exposed inside synchronous dispatch while its owning mutex is locked;
// no Rust reference or Objective-C object escapes that operation. CoreML arrays
// have no thread affinity. Moving an idle owner between threads is safe; sharing
// mutable access without the mutex is deliberately not supported (no Sync impl).
unsafe impl Send for NativeTensor {}

impl Drop for NativeTensor {
    fn drop(&mut self) {
        unsafe {
            if !self.view.is_null() {
                let _: () = msg_send![self.view, release];
            }
            dealloc(self.data.as_ptr(), self.layout);
        }
    }
}

impl NativeTensor {
    fn new(dtype: DataType, capacity: usize) -> Result<Self, GraphError> {
        // Supported Apple platforms have 4 or 16 KiB pages. This alignment also
        // meets CoreML's outputBackings recommendation on both.
        let layout = Layout::from_size_align(capacity.max(1), 16384)
            .map_err(|e| failed(format!("native tensor allocation: {e}")))?;
        let data = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| failed("native tensor allocation failed"))?;
        Ok(Self {
            data,
            layout,
            capacity,
            dtype,
            view: ptr::null_mut(),
            shape: Vec::new(),
        })
    }

    fn bytes(&self, len: usize) -> Result<&[u8], GraphError> {
        if len > self.capacity {
            return Err(failed("tensor storage shorter than logical size"));
        }
        Ok(unsafe { std::slice::from_raw_parts(self.data.as_ptr(), len) })
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), GraphError> {
        if bytes.len() > self.capacity {
            return Err(failed("write exceeds tensor capacity"));
        }
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.as_ptr(), bytes.len()) };
        Ok(())
    }

    fn array(&mut self, descriptor: &OperandDescriptor) -> Result<*mut Object, GraphError> {
        let shape = physical_shape(descriptor)?;
        let length = descriptor
            .byte_length()
            .ok_or_else(|| failed("tensor byte length overflow"))?;
        if length > self.capacity || descriptor.data_type != self.dtype {
            return Err(failed("native tensor shape/type exceeds owned storage"));
        }
        if self.view.is_null() || self.shape != shape {
            let mut strides = vec![1i64; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1]
                    .checked_mul(shape[i + 1])
                    .ok_or_else(|| failed("native tensor stride overflow"))?;
            }
            let mut view = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = unsafe {
                rustnn_coreml_array_view(
                    self.data.as_ptr().cast(),
                    shape.as_ptr(),
                    strides.as_ptr(),
                    shape.len(),
                    native_code(self.dtype).expect("native dtype"),
                    &mut view,
                    error.as_mut_ptr().cast(),
                    error.len(),
                )
            };
            if status != 0 || view.is_null() {
                return Err(failed(format!(
                    "native tensor view: {}",
                    shim_error_to_string(&error)
                )));
            }
            unsafe {
                if !self.view.is_null() {
                    let _: () = msg_send![self.view, release];
                }
            }
            self.view = view;
            self.shape = shape;
        }
        Ok(self.view)
    }
}

impl CoremlTensorStorage {
    pub(crate) fn new(dtype: DataType, capacity: usize, reuse: bool) -> Result<Self, GraphError> {
        if reuse && native_code(dtype).is_some() {
            Ok(Self::Native(Mutex::new(NativeTensor::new(
                dtype, capacity,
            )?)))
        } else {
            Ok(Self::Host(vec![0; capacity.max(1)]))
        }
    }

    pub(crate) fn is_native(&self) -> bool {
        matches!(self, Self::Native(_))
    }

    pub(crate) fn host(&self) -> Result<&[u8], GraphError> {
        match self {
            Self::Host(bytes) => Ok(bytes),
            Self::Native(_) => Err(failed("expected host tensor")),
        }
    }

    pub(crate) fn read(&self, destination: &mut [u8]) -> Result<(), GraphError> {
        match self {
            Self::Host(bytes) => destination.copy_from_slice(
                bytes
                    .get(..destination.len())
                    .ok_or_else(|| failed("tensor storage shorter than logical size"))?,
            ),
            Self::Native(native) => destination.copy_from_slice(
                native
                    .lock()
                    .map_err(|_| failed("poisoned native tensor"))?
                    .bytes(destination.len())?,
            ),
        }
        Ok(())
    }

    pub(crate) fn write(&mut self, source: &[u8]) -> Result<(), GraphError> {
        match self {
            Self::Host(bytes) => bytes
                .get_mut(..source.len())
                .ok_or_else(|| failed("write exceeds tensor capacity"))?
                .copy_from_slice(source),
            Self::Native(native) => native
                .get_mut()
                .map_err(|_| failed("poisoned native tensor"))?
                .write(source)?,
        }
        Ok(())
    }

    /// Grow while preserving bytes, or replace with zeroed capacity (reserve API).
    pub(crate) fn reserve(&mut self, capacity: usize, preserve: bool) -> Result<bool, GraphError> {
        match self {
            Self::Host(bytes) => {
                if !preserve {
                    *bytes = vec![0; capacity.max(1)];
                } else if capacity > bytes.len() {
                    bytes.resize(capacity, 0);
                }
                Ok(false)
            }
            Self::Native(native) => {
                let old = native
                    .get_mut()
                    .map_err(|_| failed("poisoned native tensor"))?;
                if preserve && capacity <= old.capacity {
                    return Ok(false);
                }
                let mut replacement = NativeTensor::new(old.dtype, capacity)?;
                if preserve {
                    replacement.write(old.bytes(old.capacity.min(capacity))?)?;
                }
                *old = replacement;
                Ok(true)
            }
        }
    }
}

fn physical_shape(descriptor: &OperandDescriptor) -> Result<Vec<i64>, GraphError> {
    let shape = descriptor.static_or_max_shape();
    if shape.contains(&0) {
        return Err(failed("CoreML cannot bind a zero-extent tensor"));
    }
    Ok(if shape.is_empty() {
        vec![1]
    } else {
        shape.iter().map(|&d| i64::from(d)).collect()
    })
}

pub(crate) struct CoremlTensorBinding<'a> {
    pub(crate) storage: &'a CoremlTensorStorage,
    pub(crate) descriptor: &'a OperandDescriptor,
}

/// Execute with retained input views and optional destination backings. Native
/// destinations keep exclusive ownership even when CoreML returns aliased views.
pub(crate) fn run_coreml_tensors(
    model: &CompiledCoremlModel,
    inputs: &HashMap<String, CoremlTensorBinding<'_>>,
    outputs: &HashMap<String, CoremlTensorBinding<'_>>,
    output_backings: bool,
    statistics: &mut CoremlTensorStatistics,
) -> Result<HashMap<String, Vec<u8>>, GraphError> {
    let mut unique = HashSet::new();
    for binding in inputs.values().chain(outputs.values()) {
        if !unique.insert(ptr::from_ref(binding.storage)) {
            return Err(failed("duplicate CoreML tensor storage binding"));
        }
    }
    autoreleasepool(|| unsafe {
        let model_description: *mut Object = msg_send![model.model, modelDescription];
        let input_descs: *mut Object = msg_send![model_description, inputDescriptionsByName];
        let output_descs: *mut Object = msg_send![model_description, outputDescriptionsByName];
        let dict: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];
        let backings: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];
        let mut guards = Vec::new();
        let mut destinations = HashMap::new();
        for (name, binding) in inputs {
            let key = nsstring_from_str(name)?;
            let code = model_input_dtype_code(input_descs, key)
                .map_or_else(|| map_dtype(binding.descriptor.data_type), Ok)?;
            let logical = binding
                .descriptor
                .byte_length()
                .ok_or_else(|| failed("input size overflow"))?;
            let array = match binding.storage {
                CoremlTensorStorage::Native(native) => {
                    let mut guard = native
                        .lock()
                        .map_err(|_| failed("poisoned native tensor"))?;
                    let array = if same_type(binding.descriptor.data_type, code) {
                        statistics.native_input_bindings += 1;
                        guard.array(binding.descriptor)?
                    } else {
                        let array = create_multi_array(&physical_shape(binding.descriptor)?, code)?;
                        fill_multiarray_from_bytes(
                            array,
                            guard.bytes(logical)?,
                            binding.descriptor.data_type,
                            code,
                        )?;
                        statistics.input_copy_bytes += logical as u64;
                        array
                    };
                    guards.push(guard);
                    array
                }
                CoremlTensorStorage::Host(bytes) => {
                    let array = create_multi_array(&physical_shape(binding.descriptor)?, code)?;
                    fill_multiarray_from_bytes(
                        array,
                        bytes
                            .get(..logical)
                            .ok_or_else(|| failed("short input storage"))?,
                        binding.descriptor.data_type,
                        code,
                    )?;
                    statistics.input_copy_bytes += logical as u64;
                    array
                }
            };
            let feature: *mut Object =
                msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
            let _: () = msg_send![dict, setObject: feature forKey: key];
        }
        for (name, binding) in outputs {
            if let CoremlTensorStorage::Native(native) = binding.storage {
                let mut guard = native
                    .lock()
                    .map_err(|_| failed("poisoned native tensor"))?;
                let array = guard.array(binding.descriptor)?;
                let key = nsstring_from_str(name)?;
                let code = model_input_dtype_code(output_descs, key);
                if output_backings
                    && code.is_some_and(|code| same_type(binding.descriptor.data_type, code))
                {
                    let feature: *mut Object =
                        msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
                    let description: *mut Object = msg_send![output_descs, objectForKey: key];
                    let constraint: *mut Object = msg_send![description, multiArrayConstraint];
                    let shape_constraint: *mut Object = msg_send![constraint, shapeConstraint];
                    let constraint_type: i64 = msg_send![shape_constraint, type];
                    let allowed: bool = msg_send![description, isAllowedValue: feature];
                    // CoreML rejects outputBackings for flexible output features,
                    // even when isAllowedValue accepts this particular shape.
                    // Fixed outputs are represented as a singleton enumeration
                    // (MLMultiArrayShapeConstraintTypeEnumerated = 2). Ranged,
                    // multi-shape and unconstrained outputs use the copy fallback.
                    let enumerated_count = if constraint_type == 2 {
                        let shapes: *mut Object = msg_send![shape_constraint, enumeratedShapes];
                        msg_send![shapes, count]
                    } else {
                        0
                    };
                    if is_fixed_output_shape(constraint_type, enumerated_count) && allowed {
                        let _: () = msg_send![backings, setObject: array forKey: key];
                        statistics.output_backings_requested += 1;
                        destinations.insert(name, (guards.len(), true));
                    } else {
                        destinations.insert(name, (guards.len(), false));
                    }
                } else {
                    destinations.insert(name, (guards.len(), false));
                }
                guards.push(guard);
            }
        }
        let mut create_error: *mut Object = ptr::null_mut();
        let provider_alloc: *mut Object = msg_send![class!(MLDictionaryFeatureProvider), alloc];
        let provider: *mut Object =
            msg_send![provider_alloc, initWithDictionary: dict error: &mut create_error];
        if provider.is_null() {
            return Err(failed(ns_error_to_string(
                create_error,
                "feature provider init failed",
            )));
        }
        let _provider_guard = ReleaseOnDrop(provider);
        let mut output_provider: *mut Object = ptr::null_mut();
        let mut error = [0u8; 1024];
        let status = rustnn_coreml_predict_backed(
            model.model,
            provider,
            backings,
            &mut output_provider,
            error.as_mut_ptr().cast(),
            error.len(),
        );
        if status != 0 || output_provider.is_null() {
            return Err(failed(format!(
                "prediction failed: {}",
                shim_error_to_string(&error)
            )));
        }
        let _output_guard = ReleaseOnDrop(output_provider);
        let mut result = HashMap::new();
        for (name, binding) in outputs {
            let key = nsstring_from_str(name)?;
            let feature: *mut Object = msg_send![output_provider, featureValueForName: key];
            if feature.is_null() {
                return Err(failed(format!("missing output '{name}'")));
            }
            let array: *mut Object = msg_send![feature, multiArrayValue];
            if array.is_null() {
                return Err(failed(format!("output '{name}' is not a tensor")));
            }
            let actual_shape: *mut Object = msg_send![array, shape];
            let actual_shape = nsarray_to_i64_vec(actual_shape)?;
            let expected_shape = physical_shape(binding.descriptor)?;
            if actual_shape != expected_shape {
                return Err(failed(format!(
                    "output '{name}': actual shape {actual_shape:?}, expected {expected_shape:?}"
                )));
            }
            let logical = binding
                .descriptor
                .byte_length()
                .ok_or_else(|| failed("output size overflow"))?;
            if let Some(&(index, requested)) = destinations.get(name) {
                let destination = &mut guards[index];
                if requested && array == destination.view {
                    statistics.output_backings_accepted += 1;
                } else {
                    copy_output(array, destination, binding.descriptor)?;
                    statistics.output_copy_bytes += logical as u64;
                }
            } else {
                result.insert(
                    name.clone(),
                    extract_multiarray_bytes(array, binding.descriptor)?,
                );
                statistics.output_copy_bytes += logical as u64;
            }
        }
        Ok(result)
    })
}

unsafe fn copy_output(
    array: *mut Object,
    destination: &mut NativeTensor,
    descriptor: &OperandDescriptor,
) -> Result<(), GraphError> {
    let code: i64 = msg_send![array, dataType];
    if !same_type(descriptor.data_type, code as i32) {
        return destination.write(&unsafe { extract_multiarray_bytes(array, descriptor)? });
    }
    let shape_obj: *mut Object = msg_send![array, shape];
    let stride_obj: *mut Object = msg_send![array, strides];
    let shape = unsafe { nsarray_to_i64_vec(shape_obj)? };
    let strides = unsafe { nsarray_to_i64_vec(stride_obj)? };
    let logical = descriptor
        .byte_length()
        .ok_or_else(|| failed("output size overflow"))?;
    if logical > destination.capacity
        || shape.len() != strides.len()
        || strides.iter().any(|&s| s < 0)
    {
        return Err(failed("invalid native output layout"));
    }
    let source: *const u8 = msg_send![array, dataPointer];
    if source.is_null() {
        return Err(failed("null native output buffer"));
    }
    if is_contiguous(&shape, &strides) {
        // Overlap is legal if CoreML returns another view of the proposed backing.
        unsafe { ptr::copy(source, destination.data.as_ptr(), logical) };
    } else {
        let element = descriptor.data_type.bytes_per_element();
        let mut maximum = 0usize;
        for (&size, &stride) in shape.iter().zip(&strides) {
            let span = usize::try_from(size - 1)
                .ok()
                .and_then(|s| s.checked_mul(stride as usize))
                .ok_or_else(|| failed("output stride overflow"))?;
            maximum = maximum
                .checked_add(span)
                .ok_or_else(|| failed("output offset overflow"))?;
        }
        if maximum
            .checked_add(1)
            .and_then(|n| n.checked_mul(element))
            .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(failed("output offset exceeds addressable storage"));
        }
        let source_end = source
            .addr()
            .checked_add((maximum + 1) * element)
            .ok_or_else(|| failed("output address overflow"))?;
        let destination_end = destination
            .data
            .as_ptr()
            .addr()
            .checked_add(logical)
            .ok_or_else(|| failed("destination address overflow"))?;
        if source.addr() < destination_end && destination.data.as_ptr().addr() < source_end {
            // Per-element memmove is insufficient for an overlapping transpose:
            // an early destination write can destroy a later source element.
            let gathered = unsafe { gather_strided_bytes(source, &shape, &strides, element) };
            return destination.write(&gathered);
        }
        // Distinct returned arrays may alias input buffers, so never retain them
        // as a logical output tensor. Gather into this tensor's private allocation.
        for index in 0..logical / element {
            let mut remaining = index;
            let mut offset = 0usize;
            for axis in (0..shape.len()).rev() {
                offset += (remaining % shape[axis] as usize) * strides[axis] as usize;
                remaining /= shape[axis] as usize;
            }
            unsafe {
                ptr::copy(
                    source.add(offset * element),
                    destination.data.as_ptr().add(index * element),
                    element,
                )
            };
        }
    }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos"))]
unsafe extern "C" {
    fn rustnn_coreml_array_view(
        data: *mut c_void,
        shape: *const i64,
        strides: *const i64,
        rank: usize,
        dtype: i32,
        out: *mut *mut Object,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn rustnn_coreml_predict_backed(
        model: *mut Object,
        features: *mut Object,
        backings: *mut Object,
        out: *mut *mut Object,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
}

#[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "tvos")))]
#[allow(clippy::too_many_arguments)] // Mirrors the native C ABI above.
unsafe fn rustnn_coreml_array_view(
    _data: *mut c_void,
    _shape: *const i64,
    _strides: *const i64,
    _rank: usize,
    _dtype: i32,
    _out: *mut *mut Object,
    _error: *mut c_char,
    _length: usize,
) -> i32 {
    1
}
#[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "tvos")))]
unsafe fn rustnn_coreml_predict_backed(
    _model: *mut Object,
    _features: *mut Object,
    _backings: *mut Object,
    _out: *mut *mut Object,
    _error: *mut c_char,
    _length: usize,
) -> i32 {
    1
}

#[cfg(test)]
mod shape_tests {
    use super::is_fixed_output_shape;

    #[test]
    fn output_backings_require_a_single_enumerated_shape() {
        assert!(is_fixed_output_shape(2, 1));
        for constraint_type in [0, 1, 3] {
            assert!(!is_fixed_output_shape(constraint_type, 1));
        }
        for count in [0, 2, 3] {
            assert!(!is_fixed_output_shape(2, count));
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn descriptor(shape: &[u32]) -> OperandDescriptor {
        OperandDescriptor {
            data_type: DataType::Float32,
            shape: crate::graph::to_dimension_vector(shape),
            pending_permutation: vec![],
        }
    }

    #[test]
    fn fixed_output_backing_proposals_follow_loaded_metadata() {
        use crate::backend_selection::DeviceType;
        use crate::converters::{CoremlMlProgramConverter, GraphConverter};
        use crate::graph::{GraphInfo, Operand, OperandKind};
        use crate::operators::Operation;

        let descriptor = descriptor(&[3]);
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    name: Some("input".into()),
                    kind: OperandKind::Input,
                    descriptor: descriptor.clone(),
                },
                Operand {
                    name: Some("result".into()),
                    kind: OperandKind::Output,
                    descriptor: descriptor.clone(),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation::Relu {
                input: 0,
                options: None,
                outputs: vec![1],
            }],
            ..Default::default()
        };
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let model = compile_model(
            converted.data,
            converted.weights_data,
            DeviceType::Cpu,
            false,
        )
        .unwrap();
        let mut input = CoremlTensorStorage::new(DataType::Float32, 12, true).unwrap();
        let output = CoremlTensorStorage::new(DataType::Float32, 12, true).unwrap();
        let (eligible, dtype, constraint_type, shape_count, allowed) = autoreleasepool(|| unsafe {
            let model_description: *mut Object = msg_send![model.model, modelDescription];
            let descriptions: *mut Object = msg_send![model_description, outputDescriptionsByName];
            let key = nsstring_from_str("result").unwrap();
            let description: *mut Object = msg_send![descriptions, objectForKey: key];
            let constraint: *mut Object = msg_send![description, multiArrayConstraint];
            assert!(!constraint.is_null(), "result must have array metadata");
            let dtype: i64 = msg_send![constraint, dataType];
            let shape_constraint: *mut Object = msg_send![constraint, shapeConstraint];
            let constraint_type: i64 = msg_send![shape_constraint, type];
            let shape_count: usize = if constraint_type == 2 {
                let shapes: *mut Object = msg_send![shape_constraint, enumeratedShapes];
                msg_send![shapes, count]
            } else {
                0
            };
            let CoremlTensorStorage::Native(native) = &output else {
                panic!("native output storage");
            };
            let mut guard = native.lock().unwrap();
            let array = guard.array(&descriptor).unwrap();
            let feature: *mut Object =
                msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
            let allowed: bool = msg_send![description, isAllowedValue: feature];
            // Inspect the loaded model independently of the production predicate.
            // Metadata can differ by runtime; an ineligible output must still copy.
            let eligible =
                matches!(dtype, 32 | 65568) && constraint_type == 2 && shape_count == 1 && allowed;
            (eligible, dtype, constraint_type, shape_count, allowed)
        });
        let mut statistics = CoremlTensorStatistics::default();
        for _ in 0..4 {
            input.write(bytemuck::cast_slice(&[-1f32, 2., 3.])).unwrap();
            let bind = |storage| CoremlTensorBinding {
                storage,
                descriptor: &descriptor,
            };
            let result = run_coreml_tensors(
                &model,
                &HashMap::from([("input".into(), bind(&input))]),
                &HashMap::from([("result".into(), bind(&output))]),
                true,
                &mut statistics,
            )
            .unwrap();
            assert!(result.is_empty());
            // Mutation after prediction must not change the logical output.
            input.write(bytemuck::cast_slice(&[99f32; 3])).unwrap();
            let mut actual = [0u8; 12];
            output.read(&mut actual).unwrap();
            assert_eq!(
                actual.as_slice(),
                bytemuck::cast_slice::<f32, u8>(&[0f32, 2., 3.])
            );
        }
        assert_eq!(
            statistics.output_backings_requested,
            if eligible { 4 } else { 0 },
            "dtype={dtype}, constraint_type={constraint_type}, shapes={shape_count}, allowed={allowed}"
        );
        assert!(statistics.output_backings_accepted <= statistics.output_backings_requested);
        assert_eq!(
            statistics.output_copy_bytes,
            (4 - statistics.output_backings_accepted) * 12
        );
    }

    #[test]
    fn native_capacity_preserves_bytes_and_rebuilds_active_views() {
        autoreleasepool(|| {
            let mut storage = CoremlTensorStorage::new(DataType::Float32, 8, true).unwrap();
            storage.write(bytemuck::cast_slice(&[1.0f32, 2.0])).unwrap();
            let CoremlTensorStorage::Native(native) = &storage else {
                panic!("native storage");
            };
            {
                let mut guard = native.lock().unwrap();
                assert_eq!(guard.data.as_ptr().addr() % 16384, 0);
                let first = guard.array(&descriptor(&[2])).unwrap();
                assert_eq!(first, guard.array(&descriptor(&[2])).unwrap());
            }
            assert!(storage.reserve(16, true).unwrap());
            let mut bytes = [0u8; 16];
            storage.read(&mut bytes).unwrap();
            assert_eq!(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .copied()
                    .map(f32::from_le_bytes)
                    .collect::<Vec<_>>(),
                [1.0, 2.0, 0.0, 0.0]
            );
            assert!(!storage.reserve(4, true).unwrap());
            let CoremlTensorStorage::Native(native) = &storage else {
                panic!("native storage");
            };
            let mut guard = native.lock().unwrap();
            for shape in [&[4][..], &[2, 2], &[1], &[]] {
                guard.array(&descriptor(shape)).unwrap();
                assert_eq!(guard.shape, physical_shape(&descriptor(shape)).unwrap());
            }
            assert!(guard.array(&descriptor(&[0])).is_err());
        });
    }

    #[test]
    fn strided_output_alias_is_gathered_before_destination_is_written() {
        autoreleasepool(|| {
            let mut destination = NativeTensor::new(DataType::Float32, 16).unwrap();
            destination
                .write(bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]))
                .unwrap();
            let mut view = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = unsafe {
                rustnn_coreml_array_view(
                    destination.data.as_ptr().cast(),
                    [2, 2].as_ptr(),
                    [1, 2].as_ptr(),
                    2,
                    65568,
                    &mut view,
                    error.as_mut_ptr().cast(),
                    error.len(),
                )
            };
            assert_eq!(status, 0, "{}", shim_error_to_string(&error));
            let _view_guard = ReleaseOnDrop(view);
            unsafe {
                copy_output(view, &mut destination, &descriptor(&[2, 2])).unwrap();
            }
            assert_eq!(
                bytemuck::cast_slice::<u8, f32>(destination.bytes(16).unwrap()),
                &[1.0, 3.0, 2.0, 4.0]
            );
        });
    }

    #[test]
    fn native_tensor_storage_is_send_and_sync_through_its_mutex() {
        fn assert_traits<T: Send + Sync>() {}
        assert_traits::<CoremlTensorStorage>();
        let mut tensor = CoremlTensorStorage::new(DataType::Int32, 4, true).unwrap();
        tensor.write(&16777217i32.to_le_bytes()).unwrap();
        let bytes = std::thread::spawn(move || {
            let mut bytes = [0; 4];
            tensor.read(&mut bytes).unwrap();
            bytes
        })
        .join()
        .unwrap();
        assert_eq!(i32::from_le_bytes(bytes), 16777217);
    }
}
