pub use crate::tensor_proto::DataType;

/// ONNX tensor data types
impl DataType {
    pub(crate) fn from_onnx_type(data_type: i32) -> Self {
        Self::try_from(data_type).unwrap_or(Self::Undefined)
    }

    /// Get the minimum storage size in bytes required to hold a single element.
    ///
    /// Sub-byte packed types (`Uint4`, `Int4`, `Float4e2m1`, `Uint2`, `Int2`) return `Some(1)`,
    /// representing the minimum byte container per standalone element (note that in ONNX tensor
    /// buffers, multiple sub-byte elements are packed into each byte).
    ///
    /// Returns `None` for dynamically-sized or undefined types (`String` and `Undefined`).
    #[must_use]
    pub const fn size_in_bytes(self) -> Option<usize> {
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

    /// Check if this is a floating-point type
    #[must_use]
    pub const fn is_float(self) -> bool {
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
    #[must_use]
    pub const fn is_integer(self) -> bool {
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
