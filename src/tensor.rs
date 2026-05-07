use prost::bytes::Bytes;
use std::borrow::Cow;
use std::{mem, slice};

use crate::{
    DataType, Error, external_data::ExternalDataInfo, tensor_shape_proto::dimension::Value,
    type_proto::Tensor,
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

/// Zero-copy tensor data
#[derive(Debug, Clone)]
pub enum TensorData<'a> {
    /// Contiguous buffer (mmap-backed or loaded)
    Raw(Bytes),
    /// String tensor elements, each Arc-backed
    Strings(Cow<'a, [Bytes]>),
    /// Typed numeric data (memory taken from TensorProto or borrowed from model)
    F32(Cow<'a, [f32]>),
    F64(Cow<'a, [f64]>),
    I32(Cow<'a, [i32]>),
    I64(Cow<'a, [i64]>),
    U64(Cow<'a, [u64]>),
}

impl<'a> TensorData<'a> {
    /// Total byte length across all variants
    ///
    /// For Strings, returns sum of all string element bytes.
    /// If all Strings are empty, returns 0.
    pub fn len(&self) -> usize {
        match self {
            TensorData::Raw(b) => b.len(),
            TensorData::Strings(parts) => parts.iter().map(|b| b.len()).sum(),
            TensorData::F32(v) => v.len() * mem::size_of::<f32>(),
            TensorData::F64(v) => v.len() * mem::size_of::<f64>(),
            TensorData::I32(v) => v.len() * mem::size_of::<i32>(),
            TensorData::I64(v) => v.len() * mem::size_of::<i64>(),
            TensorData::U64(v) => v.len() * mem::size_of::<u64>(),
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

    /// Convert to owned data that can outlive the tensor
    ///
    /// This consumes self and returns data with no borrowed references.
    /// Note that Numeric data will be copied if borrowed. For zero-copy owned data,
    /// use OnnxTensor::into_data instead.
    pub fn into_owned(self) -> TensorData<'static> {
        match self {
            TensorData::Raw(b) => TensorData::Raw(b),
            TensorData::Strings(s) => TensorData::Strings(Cow::Owned(s.into_owned())),
            TensorData::F32(v) => TensorData::F32(Cow::Owned(v.into_owned())),
            TensorData::F64(v) => TensorData::F64(Cow::Owned(v.into_owned())),
            TensorData::I32(v) => TensorData::I32(Cow::Owned(v.into_owned())),
            TensorData::I64(v) => TensorData::I64(Cow::Owned(v.into_owned())),
            TensorData::U64(v) => TensorData::U64(Cow::Owned(v.into_owned())),
        }
    }

    /// Get data as contiguous byte slice
    ///
    /// Raw and Numeric variants borrow directly. Strings with single element borrows,
    /// multiple elements concatenate into owned Vec.
    pub fn as_slice(&self) -> Cow<'_, [u8]> {
        match self {
            TensorData::Raw(b) => Cow::Borrowed(b.as_ref()),
            TensorData::F32(v) => Cow::Borrowed(slice_as_u8(v)),
            TensorData::F64(v) => Cow::Borrowed(slice_as_u8(v)),
            TensorData::I32(v) => Cow::Borrowed(slice_as_u8(v)),
            TensorData::I64(v) => Cow::Borrowed(slice_as_u8(v)),
            TensorData::U64(v) => Cow::Borrowed(slice_as_u8(v)),
            TensorData::Strings(s) => {
                if s.len() == 1 {
                    Cow::Borrowed(s[0].as_ref())
                } else if s.is_empty() {
                    Cow::Borrowed(&[])
                } else {
                    let total = s.iter().map(|bytes| bytes.len()).sum();
                    let mut vec = Vec::with_capacity(total);
                    for bytes in s.as_ref() {
                        vec.extend_from_slice(bytes);
                    }
                    Cow::Owned(vec)
                }
            }
        }
    }
}

/// Information about an ONNX tensor
#[derive(Debug)]
pub struct OnnxTensor {
    name: String,
    shape: Vec<i64>,
    data_type: DataType,
    data: TensorDataLocation,
}

impl OnnxTensor {
    pub(crate) fn new(
        name: String,
        shape: Vec<i64>,
        data_type: DataType,
        data: TensorDataLocation,
    ) -> Self {
        OnnxTensor {
            name,
            shape,
            data_type,
            data,
        }
    }

    pub(crate) fn from_tensor_type(name: String, tensor_type: &Tensor) -> Result<Self, Error> {
        let shape = if let Some(shape_proto) = &tensor_type.shape {
            shape_proto
                .dim
                .iter()
                .map(|d| match &d.value {
                    Some(Value::DimValue(v)) => *v,
                    _ => -1,
                })
                .collect()
        } else {
            Vec::new()
        };

        let elem_type = tensor_type
            .elem_type
            .ok_or_else(|| Error::MissingField("tensor elem_type".to_string()))?;
        if elem_type == 0 {
            return Err(Error::InvalidModel(
                "tensor elem_type must not be UNDEFINED (0)".to_string(),
            ));
        }

        Ok(OnnxTensor::new(
            name,
            shape,
            DataType::from_onnx_type(elem_type),
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

    /// Borrow tensor data
    ///
    /// Raw and Strings variants clone Arc pointers only, Numeric borrows directly.
    pub fn data(&self) -> Result<TensorData<'_>, Error> {
        match &self.data {
            TensorDataLocation::External(external_info) => {
                Ok(TensorData::Raw(external_info.load_data()?))
            }
            TensorDataLocation::Mmap(bytes) => Ok(TensorData::Raw(bytes.clone())),
            TensorDataLocation::MmapStrings(strings) => {
                Ok(TensorData::Strings(Cow::Borrowed(strings)))
            }
            TensorDataLocation::F32(v) => Ok(TensorData::F32(Cow::Borrowed(v))),
            TensorDataLocation::F64(v) => Ok(TensorData::F64(Cow::Borrowed(v))),
            TensorDataLocation::I64(v) => Ok(TensorData::I64(Cow::Borrowed(v))),
            TensorDataLocation::U64(v) => Ok(TensorData::U64(Cow::Borrowed(v))),
            TensorDataLocation::I32(v) => Ok(TensorData::I32(Cow::Borrowed(v))),
            TensorDataLocation::None => Err(Error::MissingField("tensor data".to_string())),
        }
    }

    /// Consume tensor and return owned data
    ///
    /// All variants returned with no borrowed references
    pub fn into_data(self) -> Result<TensorData<'static>, Error> {
        match self.data {
            TensorDataLocation::External(external_info) => {
                Ok(TensorData::Raw(external_info.load_data()?))
            }
            TensorDataLocation::Mmap(bytes) => Ok(TensorData::Raw(bytes)),
            TensorDataLocation::MmapStrings(strings) => {
                Ok(TensorData::Strings(Cow::Owned(strings)))
            }
            TensorDataLocation::F32(v) => Ok(TensorData::F32(Cow::Owned(v))),
            TensorDataLocation::F64(v) => Ok(TensorData::F64(Cow::Owned(v))),
            TensorDataLocation::I64(v) => Ok(TensorData::I64(Cow::Owned(v))),
            TensorDataLocation::U64(v) => Ok(TensorData::U64(Cow::Owned(v))),
            TensorDataLocation::I32(v) => Ok(TensorData::I32(Cow::Owned(v))),
            TensorDataLocation::None => Err(Error::MissingField("tensor data".to_string())),
        }
    }
}

fn slice_as_u8<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(slice.as_ptr() as *const u8, mem::size_of_val(slice)) }
}
