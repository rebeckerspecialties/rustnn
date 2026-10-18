//! Shape metadata belongs to Model.description, independently of MIL types.
//!
//! The loader receives the original description bytes. This view supplies
//! MIL's FlexibleShapeInformation annotation and the human-readable schema.

use crate::pb_reader::{PBReader, read_packed_varints};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ShapeFlexibility {
    Ranges(Vec<(u64, i64)>),
    Enumerated(Vec<Vec<i64>>),
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FeatureDescription {
    pub name: String,
    pub short_description: String,
    pub data_type: u64,
    pub optional: bool,
    pub shape: Vec<i64>,
    pub flexibility: Option<ShapeFlexibility>,
}

impl FeatureDescription {
    pub fn default_shape(&self) -> Vec<i64> {
        if !self.shape.is_empty() {
            return self.shape.clone();
        }
        match &self.flexibility {
            Some(ShapeFlexibility::Ranges(ranges)) => {
                ranges.iter().map(|(min, _)| *min as i64).collect()
            }
            Some(ShapeFlexibility::Enumerated(shapes)) => {
                shapes.first().cloned().unwrap_or_default()
            }
            None => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FunctionDescription {
    pub name: String,
    pub inputs: Vec<FeatureDescription>,
    pub outputs: Vec<FeatureDescription>,
}

#[derive(Debug, Default)]
pub(crate) struct ModelDescription {
    pub main: FunctionDescription,
    pub functions: Vec<FunctionDescription>,
    pub default_function: String,
    pub has_metadata: bool,
    pub user_defined_metadata: BTreeMap<String, String>,
}

impl ModelDescription {
    pub fn decode(data: &[u8]) -> Self {
        let mut reader = PBReader::new(data);
        let mut result = Self::default();
        while let Some((field, wire)) = reader.read_tag() {
            match field {
                1 => result
                    .main
                    .inputs
                    .push(decode_feature(reader.read_length_delimited())),
                10 => result
                    .main
                    .outputs
                    .push(decode_feature(reader.read_length_delimited())),
                20 => result
                    .functions
                    .push(decode_function(reader.read_length_delimited())),
                21 => result.default_function = reader.read_string(),
                100 => {
                    result.has_metadata = true;
                    decode_metadata(reader.read_length_delimited(), &mut result);
                }
                _ => reader.skip(wire),
            }
        }
        result
    }

    pub fn function(&self, name: &str) -> Option<&FunctionDescription> {
        if self.functions.is_empty() {
            (name == "main").then_some(&self.main)
        } else {
            self.functions.iter().find(|function| function.name == name)
        }
    }
}

fn decode_metadata(data: &[u8], description: &mut ModelDescription) {
    let mut reader = PBReader::new(data);
    while let Some((field, wire)) = reader.read_tag() {
        if field == 100 {
            // Metadata.userDefined is a protobuf map<string, string>. Preserve
            // generic producer metadata; interpreting it belongs to callers.
            let mut entry = PBReader::new(reader.read_length_delimited());
            let (mut key, mut value) = (String::new(), String::new());
            while let Some((field, wire)) = entry.read_tag() {
                match field {
                    1 => key = entry.read_string(),
                    2 => value = entry.read_string(),
                    _ => entry.skip(wire),
                }
            }
            description.user_defined_metadata.insert(key, value);
        } else {
            reader.skip(wire);
        }
    }
}

fn decode_function(data: &[u8]) -> FunctionDescription {
    let mut reader = PBReader::new(data);
    let mut result = FunctionDescription::default();
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => result.name = reader.read_string(),
            2 => result
                .inputs
                .push(decode_feature(reader.read_length_delimited())),
            3 => result
                .outputs
                .push(decode_feature(reader.read_length_delimited())),
            _ => reader.skip(wire),
        }
    }
    result
}

fn decode_feature(data: &[u8]) -> FeatureDescription {
    let mut reader = PBReader::new(data);
    let mut result = FeatureDescription::default();
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => result.name = reader.read_string(),
            2 => result.short_description = reader.read_string(),
            3 => {
                let mut ty = PBReader::new(reader.read_length_delimited());
                while let Some((field, wire)) = ty.read_tag() {
                    match field {
                        5 => decode_array(ty.read_length_delimited(), &mut result),
                        1000 => result.optional = ty.read_varint() != 0,
                        _ => ty.skip(wire),
                    }
                }
            }
            _ => reader.skip(wire),
        }
    }
    result
}

fn decode_array(data: &[u8], result: &mut FeatureDescription) {
    let mut reader = PBReader::new(data);
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 if wire == 0 => result.shape.push(reader.read_varint() as i64),
            1 => result.shape.extend(
                read_packed_varints(reader.read_length_delimited())
                    .into_iter()
                    .map(|v| v as i64),
            ),
            2 => result.data_type = reader.read_varint(),
            21 => {
                let mut enumeration = PBReader::new(reader.read_length_delimited());
                let mut shapes = Vec::new();
                while let Some((field, wire)) = enumeration.read_tag() {
                    if field == 1 {
                        let mut shape = FeatureDescription::default();
                        decode_array(enumeration.read_length_delimited(), &mut shape);
                        shapes.push(shape.shape);
                    } else {
                        enumeration.skip(wire);
                    }
                }
                result.flexibility = Some(ShapeFlexibility::Enumerated(shapes));
            }
            31 => {
                let mut range = PBReader::new(reader.read_length_delimited());
                let mut ranges = Vec::new();
                while let Some((field, wire)) = range.read_tag() {
                    if field == 1 {
                        let mut bounds = PBReader::new(range.read_length_delimited());
                        let (mut lower, mut upper) = (0, 0);
                        while let Some((field, wire)) = bounds.read_tag() {
                            match field {
                                1 => lower = bounds.read_varint(),
                                2 => upper = bounds.read_varint() as i64,
                                _ => bounds.skip(wire),
                            }
                        }
                        ranges.push((lower, upper));
                    } else {
                        range.skip(wire);
                    }
                }
                result.flexibility = Some(ShapeFlexibility::Ranges(ranges));
            }
            _ => reader.skip(wire),
        }
    }
}

pub(crate) fn shape_text<T: std::fmt::Display>(shape: &[T]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_and_unpacked_default_shapes_match() {
        let mut packed = FeatureDescription::default();
        let mut unpacked = FeatureDescription::default();
        decode_array(&[10, 2, 2, 4], &mut packed);
        decode_array(&[8, 2, 8, 4], &mut unpacked);
        assert_eq!(packed.default_shape(), vec![2, 4]);
        assert_eq!(packed.default_shape(), unpacked.default_shape());
    }

    #[test]
    fn omitted_default_uses_lower_bounds_and_keeps_unbounded_upper() {
        let mut feature = FeatureDescription::default();
        // ArrayFeatureType.shapeRange.sizeRanges = [{ lowerBound: 1, upperBound: -1 }].
        decode_array(
            &[
                0xfa, 1, 15, 10, 13, 8, 1, 16, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                0xff, 1,
            ],
            &mut feature,
        );
        assert_eq!(feature.default_shape(), vec![1]);
        assert_eq!(
            feature.flexibility,
            Some(ShapeFlexibility::Ranges(vec![(1, -1)]))
        );
    }
}
