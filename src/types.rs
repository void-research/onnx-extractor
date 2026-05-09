use crate::tensor::OnnxTensor;
use prost::bytes::Bytes;

pub use crate::tensor_proto::DataType;

/// ONNX tensor data types
impl DataType {
    /// Create DataType from ONNX type integer
    pub fn from_onnx_type(data_type: i32) -> Self {
        Self::try_from(data_type).unwrap_or(Self::Undefined)
    }

    /// Get the size in bytes for numeric types
    pub fn size_in_bytes(&self) -> Option<usize> {
        match self {
            DataType::Complex128 => Some(16),
            DataType::Double | DataType::Int64 | DataType::Uint64 | DataType::Complex64 => Some(8),
            DataType::Float | DataType::Int32 | DataType::Uint32 => Some(4),
            DataType::Float16 | DataType::Bfloat16 | DataType::Int16 | DataType::Uint16 => Some(2),
            DataType::Int8
            | DataType::Uint8
            | DataType::Bool
            | DataType::Float8e4m3fn
            | DataType::Float8e4m3fnuz
            | DataType::Float8e5m2
            | DataType::Float8e5m2fnuz
            | DataType::Float8e8m0
            | DataType::Uint4
            | DataType::Int4
            | DataType::Float4e2m1
            | DataType::Uint2
            | DataType::Int2 => Some(1),
            DataType::String | DataType::Undefined => None,
        }
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DataType::Float16
                | DataType::Float
                | DataType::Double
                | DataType::Bfloat16
                | DataType::Float8e4m3fn
                | DataType::Float8e4m3fnuz
                | DataType::Float8e5m2
                | DataType::Float8e5m2fnuz
                | DataType::Float8e8m0
                | DataType::Float4e2m1
        )
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Uint8
                | DataType::Uint16
                | DataType::Uint32
                | DataType::Uint64
                | DataType::Uint4
                | DataType::Int4
                | DataType::Uint2
                | DataType::Int2
        )
    }
}

/// ONNX attribute values
#[derive(Debug)]
pub enum AttributeValue {
    Int(i64),
    Float(f32),
    String(Bytes),
    Tensor(Box<OnnxTensor>),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Strings(Vec<Bytes>),
}

impl AttributeValue {
    /// Try to get integer value
    pub fn as_int(&self) -> Option<i64> {
        match self {
            AttributeValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to get float value
    pub fn as_float(&self) -> Option<f32> {
        match self {
            AttributeValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Try to get string value.
    ///
    /// This performs UTF-8 validation. Use `as_string_bytes` if you want to
    /// avoid validation and compare bytes directly.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            AttributeValue::String(s) => std::str::from_utf8(s).ok(),
            _ => None,
        }
    }

    /// Try to get raw bytes for a string attribute
    pub fn as_string_bytes(&self) -> Option<&[u8]> {
        match self {
            AttributeValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get tensor value
    pub fn as_tensor(&self) -> Option<&OnnxTensor> {
        match self {
            AttributeValue::Tensor(t) => Some(t.as_ref()),
            _ => None,
        }
    }

    /// Try to get integer array value
    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            AttributeValue::Ints(ints) => Some(ints),
            _ => None,
        }
    }

    /// Try to get float array value
    pub fn as_floats(&self) -> Option<&[f32]> {
        match self {
            AttributeValue::Floats(floats) => Some(floats),
            _ => None,
        }
    }

    /// Try to get string array value as validated `&str` entries.
    ///
    /// This performs UTF-8 validation on each entry and collects them into a Vec.
    /// Use `as_strings_bytes` if you want to avoid validation and allocation.
    pub fn as_strings(&self) -> Option<Vec<&str>> {
        match self {
            AttributeValue::Strings(strings) => strings
                .iter()
                .map(|s| std::str::from_utf8(s).ok())
                .collect(),
            _ => None,
        }
    }

    /// Try to get string array value as a slice of `Bytes`
    pub fn as_strings_bytes(&self) -> Option<&[Bytes]> {
        match self {
            AttributeValue::Strings(strings) => Some(strings),
            _ => None,
        }
    }
}
