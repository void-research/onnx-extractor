use prost::bytes::Bytes;
use std::borrow::Cow;
use std::{mem, slice};

use crate::{
    DataType, Error, external_data::ExternalDataInfo, tensor_shape_proto::dimension::Value,
    type_proto::Tensor,
};

#[derive(Debug, Clone)]
pub(crate) enum TensorDataLocation {
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

/// Container that safely preserves typed vector memory to avoid Undefined Behavior
/// during deallocation, while providing a uniform `[u8]` interface.
#[derive(Debug, Clone)]
pub enum NumericData<'a> {
    /// A read-only reference, usually created via `OnnxTensor::data()`.
    Borrowed(&'a [u8]),
    U8(Vec<u8>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
}

impl<'a> NumericData<'a> {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(b) => b,
            Self::U8(v) => v.as_slice(),
            Self::F32(v) => slice_as_u8(v),
            Self::F64(v) => slice_as_u8(v),
            Self::I32(v) => slice_as_u8(v),
            Self::I64(v) => slice_as_u8(v),
            Self::U64(v) => slice_as_u8(v),
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// Converts to a static owned lifetime.
    ///
    /// If this holds a `Borrowed` reference, it performs a deep copy.
    /// If it already holds an owned vector, ownership is transferred.
    pub fn into_owned(self) -> NumericData<'static> {
        match self {
            Self::Borrowed(b) => NumericData::U8(b.to_vec()),
            Self::U8(v) => NumericData::U8(v),
            Self::F32(v) => NumericData::F32(v),
            Self::F64(v) => NumericData::F64(v),
            Self::I32(v) => NumericData::I32(v),
            Self::I64(v) => NumericData::I64(v),
            Self::U64(v) => NumericData::U64(v),
        }
    }
}

/// Zero-copy tensor data
#[derive(Debug, Clone)]
pub enum TensorData<'a> {
    /// Contiguous buffer from raw_data field, Arc-backed
    Raw(Bytes),
    /// Reinterpreted numeric data from typed fields
    Numeric(NumericData<'a>),
    /// String tensor elements, each Arc-backed
    Strings(Cow<'a, [Bytes]>),
}

impl<'a> TensorData<'a> {
    /// Total byte length across all variants
    ///
    /// For Strings, returns sum of all string element bytes.
    /// If all Strings are empty, returns 0.
    pub fn len(&self) -> usize {
        match self {
            TensorData::Raw(b) => b.len(),
            TensorData::Numeric(num) => num.len(),
            TensorData::Strings(parts) => parts.iter().map(|b| b.len()).sum(),
        }
    }

    /// Returns true if data contains no elements
    ///
    /// For Raw and Numeric, equivalent to len equals zero.
    /// For Strings, checks if vector is empty. Empty strings are still elements.
    pub fn is_empty(&self) -> bool {
        match self {
            TensorData::Raw(b) => b.is_empty(),
            TensorData::Numeric(n) => n.is_empty(),
            TensorData::Strings(s) => s.is_empty(),
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
            TensorData::Numeric(n) => TensorData::Numeric(n.into_owned()),
            TensorData::Strings(s) => TensorData::Strings(Cow::Owned(s.into_owned())),
        }
    }

    /// Get data as contiguous byte slice
    ///
    /// Raw and Numeric variants borrow directly. Strings with single element borrows,
    /// multiple elements concatenate into owned Vec.
    pub fn as_slice(&self) -> Cow<'_, [u8]> {
        match self {
            TensorData::Raw(b) => Cow::Borrowed(b.as_ref()),
            TensorData::Numeric(n) => Cow::Borrowed(n.as_slice()),
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
    data_location: Option<TensorDataLocation>,
}

impl OnnxTensor {
    pub(crate) fn new(
        name: String,
        shape: Vec<i64>,
        data_type: DataType,
        data_location: Option<TensorDataLocation>,
    ) -> Self {
        OnnxTensor {
            name,
            shape,
            data_type,
            data_location,
        }
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
            None,
        ))
    }

    /// Borrow tensor data
    ///
    /// Raw and Strings variants clone Arc pointers only, Numeric borrows directly.
    /// Call into_owned on the result to detach from tensor lifetime.
    /// For external data, this lazily loads the data from the external file.
    pub fn data(&self) -> Result<TensorData<'_>, Error> {
        match &self.data_location {
            Some(TensorDataLocation::External(external_info)) => {
                Ok(TensorData::Raw(external_info.load_data()?))
            }
            Some(TensorDataLocation::Mmap(bytes)) => Ok(TensorData::Raw(bytes.clone())),
            Some(TensorDataLocation::MmapStrings(strings)) => {
                Ok(TensorData::Strings(Cow::Borrowed(strings)))
            }
            Some(TensorDataLocation::F32(v)) => Ok(TensorData::Numeric(NumericData::Borrowed(
                slice_as_u8::<f32>(v),
            ))),
            Some(TensorDataLocation::F64(v)) => Ok(TensorData::Numeric(NumericData::Borrowed(
                slice_as_u8::<f64>(v),
            ))),
            Some(TensorDataLocation::I64(v)) => Ok(TensorData::Numeric(NumericData::Borrowed(
                slice_as_u8::<i64>(v),
            ))),
            Some(TensorDataLocation::U64(v)) => Ok(TensorData::Numeric(NumericData::Borrowed(
                slice_as_u8::<u64>(v),
            ))),
            Some(TensorDataLocation::I32(v)) => Ok(TensorData::Numeric(NumericData::Borrowed(
                slice_as_u8::<i32>(v),
            ))),
            None => Err(Error::MissingField("tensor data".to_string())),
        }
    }

    /// Consume tensor and return owned data
    ///
    /// All variants returned with no borrowed references.
    /// Numeric performs zero-copy reinterpretation from typed fields.
    /// For external data, this lazily loads the data from the external file.
    pub fn into_data(mut self) -> Result<TensorData<'static>, Error> {
        match self.data_location.take() {
            Some(TensorDataLocation::External(external_info)) => {
                Ok(TensorData::Raw(external_info.load_data()?))
            }
            Some(TensorDataLocation::Mmap(bytes)) => Ok(TensorData::Raw(bytes)),
            Some(TensorDataLocation::MmapStrings(strings)) => {
                Ok(TensorData::Strings(Cow::Owned(strings)))
            }
            Some(TensorDataLocation::F32(v)) => Ok(TensorData::Numeric(NumericData::F32(v))),
            Some(TensorDataLocation::F64(v)) => Ok(TensorData::Numeric(NumericData::F64(v))),
            Some(TensorDataLocation::I64(v)) => Ok(TensorData::Numeric(NumericData::I64(v))),
            Some(TensorDataLocation::U64(v)) => Ok(TensorData::Numeric(NumericData::U64(v))),
            Some(TensorDataLocation::I32(v)) => Ok(TensorData::Numeric(NumericData::I32(v))),
            None => Err(Error::MissingField("tensor data".to_string())),
        }
    }
}

fn slice_as_u8<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(slice.as_ptr() as *const u8, mem::size_of_val(slice)) }
}
