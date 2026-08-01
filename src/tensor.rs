use prost::bytes::Bytes;
use std::{mem, slice};

use crate::{
    DataType, Error, external_data::ExternalDataInfo, tensor_shape_proto::dimension::Value,
    type_proto::Tensor as ProtoTensor,
};

#[derive(Debug, Clone)]
pub(crate) enum TensorDataLocation {
    /// No data associated with this tensor
    None,
    /// Data is stored in an external file
    External(ExternalDataInfo),
    /// Raw data as a Bytes reference (mmap-backed when loaded from file)
    Mmap(Bytes),
    /// String data as vectors of Bytes references (mmap-backed when loaded from file)
    MmapStrings(Vec<Bytes>),
    // Numeric data (memory taken from TensorProto)
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    I32(Vec<i32>),
}

/// Zero-copy tensor data reference
#[derive(Debug, Clone)]
pub enum TensorDataRef<'a> {
    /// Contiguous buffer (mmap-backed or loaded)
    Raw(Bytes),
    /// String tensor elements, each Arc-backed
    Strings(&'a [Bytes]),
    /// Typed numeric data
    F32(&'a [f32]),
    F64(&'a [f64]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    U64(&'a [u64]),
}

impl<'a> TensorDataRef<'a> {
    /// Total byte length across all variants
    ///
    /// For Strings, returns sum of all string element bytes.
    /// If all Strings are empty, returns 0.
    pub fn len(&self) -> usize {
        match self {
            TensorDataRef::Raw(b) => b.len(),
            TensorDataRef::Strings(parts) => parts.iter().map(|b| b.len()).sum(),
            TensorDataRef::F32(v) => mem::size_of_val(*v),
            TensorDataRef::F64(v) => mem::size_of_val(*v),
            TensorDataRef::I32(v) => mem::size_of_val(*v),
            TensorDataRef::I64(v) => mem::size_of_val(*v),
            TensorDataRef::U64(v) => mem::size_of_val(*v),
        }
    }

    /// Returns true if data contains no elements
    ///
    /// For Raw and Numeric, equivalent to len equals zero.
    /// For Strings, checks if vector is empty. Empty strings are still elements.
    pub fn is_empty(&self) -> bool {
        match self {
            TensorDataRef::Raw(b) => b.is_empty(),
            TensorDataRef::Strings(s) => s.is_empty(),
            TensorDataRef::F32(v) => v.is_empty(),
            TensorDataRef::F64(v) => v.is_empty(),
            TensorDataRef::I32(v) => v.is_empty(),
            TensorDataRef::I64(v) => v.is_empty(),
            TensorDataRef::U64(v) => v.is_empty(),
        }
    }

    /// Get data as contiguous byte slice
    ///
    /// Raw and Numeric variants borrow directly.
    /// Returns an error for the Strings variant as it would require concatenation and allocation.
    pub fn as_slice(&self) -> Result<&[u8], Error> {
        match self {
            TensorDataRef::Raw(b) => Ok(b),
            TensorDataRef::F32(v) => Ok(slice_as_u8(v)),
            TensorDataRef::F64(v) => Ok(slice_as_u8(v)),
            TensorDataRef::I32(v) => Ok(slice_as_u8(v)),
            TensorDataRef::I64(v) => Ok(slice_as_u8(v)),
            TensorDataRef::U64(v) => Ok(slice_as_u8(v)),
            TensorDataRef::Strings(_) => Err(Error::Unsupported(
                "String tensors cannot be accessed as a contiguous byte slice".to_string(),
            )),
        }
    }

    /// Access the string elements if the variant is Strings. Returns an error otherwise.
    pub fn strings(&self) -> Result<&'a [Bytes], Error> {
        match self {
            TensorDataRef::Strings(v) => Ok(v),
            _ => Err(Error::MissingField("tensor strings data".to_string())),
        }
    }
}

/// Zero-copy owned tensor data
#[derive(Debug, Clone)]
pub enum TensorData {
    /// Contiguous buffer (mmap-backed or loaded)
    Raw(Bytes),
    /// String tensor elements, each Arc-backed
    Strings(Vec<Bytes>),
    /// Typed numeric data
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
}

impl TensorData {
    /// Total byte length across all variants
    ///
    /// For Strings, returns sum of all string element bytes.
    /// If all Strings are empty, returns 0.
    pub fn len(&self) -> usize {
        match self {
            TensorData::Raw(b) => b.len(),
            TensorData::Strings(parts) => parts.iter().map(|b| b.len()).sum(),
            TensorData::F32(v) => mem::size_of_val(v.as_slice()),
            TensorData::F64(v) => mem::size_of_val(v.as_slice()),
            TensorData::I32(v) => mem::size_of_val(v.as_slice()),
            TensorData::I64(v) => mem::size_of_val(v.as_slice()),
            TensorData::U64(v) => mem::size_of_val(v.as_slice()),
        }
    }

    /// Returns true if data contains no elements
    ///
    /// For Raw and Numeric, equivalent to len equals zero.
    /// For Strings, checks if vector is empty. Empty strings are still elements.
    pub fn is_empty(&self) -> bool {
        match self {
            TensorData::Raw(b) => b.is_empty(),
            TensorData::Strings(s) => s.is_empty(),
            TensorData::F32(v) => v.is_empty(),
            TensorData::F64(v) => v.is_empty(),
            TensorData::I32(v) => v.is_empty(),
            TensorData::I64(v) => v.is_empty(),
            TensorData::U64(v) => v.is_empty(),
        }
    }

    /// Get data as contiguous byte slice
    ///
    /// Raw and Numeric variants borrow directly.
    /// Returns an error for the Strings variant as it would require concatenation and allocation.
    pub fn as_slice(&self) -> Result<&[u8], Error> {
        match self {
            TensorData::Raw(b) => Ok(b),
            TensorData::F32(v) => Ok(slice_as_u8(v)),
            TensorData::F64(v) => Ok(slice_as_u8(v)),
            TensorData::I32(v) => Ok(slice_as_u8(v)),
            TensorData::I64(v) => Ok(slice_as_u8(v)),
            TensorData::U64(v) => Ok(slice_as_u8(v)),
            TensorData::Strings(_) => Err(Error::Unsupported(
                "String tensors cannot be accessed as a contiguous byte slice".to_string(),
            )),
        }
    }

    /// Access the string elements if the variant is Strings. Returns an error otherwise.
    pub fn strings(&self) -> Result<&[Bytes], Error> {
        match self {
            TensorData::Strings(v) => Ok(v),
            _ => Err(Error::MissingField("tensor strings data".to_string())),
        }
    }
}

/// Information about an ONNX tensor
#[derive(Debug)]
pub struct Tensor {
    name: String,
    shape: Vec<i64>,
    data_type: DataType,
    data: TensorDataLocation,
}

impl Tensor {
    pub(crate) fn new(
        name: String,
        shape: Vec<i64>,
        data_type: DataType,
        data: TensorDataLocation,
    ) -> Self {
        Tensor {
            name,
            shape,
            data_type,
            data,
        }
    }

    pub(crate) fn from_tensor_type(name: String, tensor_type: &ProtoTensor) -> Result<Self, Error> {
        let shape = tensor_type
            .shape
            .iter()
            .flat_map(|s| &s.dim)
            .map(|d| match d.value {
                Some(Value::DimValue(v)) => v,
                _ => -1,
            })
            .collect();

        let data_type = match tensor_type.elem_type {
            Some(0) => {
                return Err(Error::InvalidModel(
                    "tensor elem_type must not be UNDEFINED (0)".to_string(),
                ));
            }
            Some(t) => DataType::from_onnx_type(t),
            None => return Err(Error::MissingField("tensor elem_type".to_string())),
        };

        Ok(Tensor::new(
            name,
            shape,
            data_type,
            TensorDataLocation::None,
        ))
    }

    /// Tensor name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Tensor shape dimensions
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Tensor data type
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns true if this tensor contains data.
    ///
    /// This check does not trigger loading or memory-mapping of external files.
    pub fn has_data(&self) -> bool {
        !matches!(self.data, TensorDataLocation::None)
    }

    /// Borrow tensor data
    ///
    /// - All variants are returned without copying the underlying tensor data.
    /// - External data is loaded from disk if not already in memory.
    pub fn data(&self) -> Result<TensorDataRef<'_>, Error> {
        match &self.data {
            TensorDataLocation::External(external_info) => {
                Ok(TensorDataRef::Raw(external_info.load_data()?))
            }
            TensorDataLocation::Mmap(bytes) => Ok(TensorDataRef::Raw(bytes.clone())),
            TensorDataLocation::MmapStrings(strings) => Ok(TensorDataRef::Strings(strings)),
            TensorDataLocation::F32(v) => Ok(TensorDataRef::F32(v)),
            TensorDataLocation::F64(v) => Ok(TensorDataRef::F64(v)),
            TensorDataLocation::I64(v) => Ok(TensorDataRef::I64(v)),
            TensorDataLocation::U64(v) => Ok(TensorDataRef::U64(v)),
            TensorDataLocation::I32(v) => Ok(TensorDataRef::I32(v)),
            TensorDataLocation::None => Err(Error::MissingField("tensor data".to_string())),
        }
    }

    /// Consume tensor and return owned data
    ///
    /// - All variants are returned without copying the underlying tensor data.
    /// - External data is loaded from disk if not already in memory.
    pub fn into_data(self) -> Result<TensorData, Error> {
        match self.data {
            TensorDataLocation::External(external_info) => {
                Ok(TensorData::Raw(external_info.load_data()?))
            }
            TensorDataLocation::Mmap(bytes) => Ok(TensorData::Raw(bytes)),
            TensorDataLocation::MmapStrings(strings) => Ok(TensorData::Strings(strings)),
            TensorDataLocation::F32(v) => Ok(TensorData::F32(v)),
            TensorDataLocation::F64(v) => Ok(TensorData::F64(v)),
            TensorDataLocation::I64(v) => Ok(TensorData::I64(v)),
            TensorDataLocation::U64(v) => Ok(TensorData::U64(v)),
            TensorDataLocation::I32(v) => Ok(TensorData::I32(v)),
            TensorDataLocation::None => Err(Error::MissingField("tensor data".to_string())),
        }
    }
}

fn slice_as_u8<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(slice.as_ptr().cast::<u8>(), mem::size_of_val(slice)) }
}
