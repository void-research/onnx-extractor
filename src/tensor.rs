use prost::bytes::Bytes;
use std::borrow::Cow;
use std::mem::ManuallyDrop;
use std::{mem, slice};

use crate::{
    DataType, Error, TensorProto, external_data::ExternalDataInfo,
    tensor_shape_proto::dimension::Value, type_proto::Tensor,
};

#[derive(Debug, Clone)]
pub(crate) enum TensorDataLocation {
    /// Data is stored in typed numeric proto fields (float_data, int32_data, etc.)
    TypedField,
    /// Data is stored in an external file
    External(ExternalDataInfo),
    /// Raw data as a Bytes reference (mmap-backed when loaded from file)
    Mmap(Bytes),
    /// String data as Vec<Bytes> references (mmap-backed when loaded from file)
    MmapStrings(Vec<Bytes>),
}

/// Zero-copy tensor data
#[derive(Debug, Clone)]
pub enum TensorData<'a> {
    /// Contiguous buffer from raw_data field, Arc-backed
    Raw(Bytes),
    /// Reinterpreted numeric data from typed fields
    Numeric(Cow<'a, [u8]>),
    /// String tensor elements, each Arc-backed
    Strings(Vec<Bytes>),
}

impl<'a> TensorData<'a> {
    /// Total byte length across all variants
    ///
    /// For Strings, returns sum of all string element bytes.
    /// If all Strings are empty, returns 0.
    pub fn len(&self) -> usize {
        match self {
            TensorData::Raw(b) => b.len(),
            TensorData::Numeric(cow) => cow.len(),
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
            TensorData::Numeric(cow) => cow.is_empty(),
            TensorData::Strings(parts) => parts.is_empty(),
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
            TensorData::Numeric(cow) => TensorData::Numeric(Cow::Owned(cow.into_owned())),
            TensorData::Strings(s) => TensorData::Strings(s),
        }
    }

    /// Get data as contiguous byte slice
    ///
    /// Raw and Numeric variants borrow directly. Strings with single element borrows,
    /// multiple elements concatenate into owned Vec.
    pub fn as_slice(&self) -> Cow<'_, [u8]> {
        match self {
            TensorData::Raw(b) => Cow::Borrowed(b.as_ref()),
            TensorData::Numeric(cow) => Cow::Borrowed(cow.as_ref()),
            TensorData::Strings(parts) => {
                if parts.len() == 1 {
                    Cow::Borrowed(parts[0].as_ref())
                } else if parts.is_empty() {
                    Cow::Borrowed(&[])
                } else {
                    let total: usize = parts.iter().map(|b| b.len()).sum();
                    let mut vec = Vec::with_capacity(total);
                    for b in parts {
                        vec.extend_from_slice(b);
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
    proto: Option<TensorProto>,
    data_location: Option<TensorDataLocation>,
}

impl OnnxTensor {
    pub(crate) fn new(
        name: String,
        shape: Vec<i64>,
        data_type: DataType,
        proto: Option<TensorProto>,
        data_location: Option<TensorDataLocation>,
    ) -> Self {
        OnnxTensor {
            name,
            shape,
            data_type,
            proto,
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
                return Ok(TensorData::Raw(external_info.load_data()?));
            }
            Some(TensorDataLocation::Mmap(bytes)) => {
                return Ok(TensorData::Raw(bytes.clone()));
            }
            Some(TensorDataLocation::MmapStrings(strings)) => {
                return Ok(TensorData::Strings(strings.clone()));
            }
            Some(TensorDataLocation::TypedField) | None => {}
        }

        // Typed field data
        let t = self
            .proto
            .as_ref()
            .ok_or_else(|| Error::MissingField("tensor data".to_string()))?;

        if let Some(raw) = &t.raw_data
            && !raw.is_empty()
        {
            return Ok(TensorData::Raw(raw.clone()));
        }

        match storage_backing(self.data_type) {
            Some(StorageBacking::F32) => {
                Ok(TensorData::Numeric(Cow::Borrowed(slice_bytes_as::<f32>(
                    t.float_data.as_slice(),
                ))))
            }
            Some(StorageBacking::F64) => {
                Ok(TensorData::Numeric(Cow::Borrowed(slice_bytes_as::<f64>(
                    t.double_data.as_slice(),
                ))))
            }
            Some(StorageBacking::I64) => {
                Ok(TensorData::Numeric(Cow::Borrowed(slice_bytes_as::<i64>(
                    t.int64_data.as_slice(),
                ))))
            }
            Some(StorageBacking::U64) => {
                Ok(TensorData::Numeric(Cow::Borrowed(slice_bytes_as::<u64>(
                    t.uint64_data.as_slice(),
                ))))
            }
            Some(StorageBacking::I32) => {
                Ok(TensorData::Numeric(Cow::Borrowed(slice_bytes_as::<i32>(
                    t.int32_data.as_slice(),
                ))))
            }
            Some(StorageBacking::Strings) => {
                if t.string_data.is_empty() && self.shape.iter().any(|&d| d != 0) {
                    return Err(Error::MissingField("tensor data".to_string()));
                }
                Ok(TensorData::Strings(t.string_data.clone()))
            }
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
                return Ok(TensorData::Raw(external_info.load_data()?));
            }
            Some(TensorDataLocation::Mmap(bytes)) => {
                return Ok(TensorData::Raw(bytes));
            }
            Some(TensorDataLocation::MmapStrings(strings)) => {
                return Ok(TensorData::Strings(strings));
            }
            Some(TensorDataLocation::TypedField) | None => {}
        }

        // Typed field data
        let t = self
            .proto
            .as_mut()
            .ok_or_else(|| Error::MissingField("tensor data".to_string()))?;

        if let Some(raw) = t.raw_data.take()
            && !raw.is_empty()
        {
            return Ok(TensorData::Raw(raw));
        }

        match storage_backing(self.data_type) {
            Some(StorageBacking::F32) => Ok(TensorData::Numeric(Cow::Owned(into_vec_u8::<f32>(
                mem::take(&mut t.float_data),
            )))),
            Some(StorageBacking::F64) => Ok(TensorData::Numeric(Cow::Owned(into_vec_u8::<f64>(
                mem::take(&mut t.double_data),
            )))),
            Some(StorageBacking::I64) => Ok(TensorData::Numeric(Cow::Owned(into_vec_u8::<i64>(
                mem::take(&mut t.int64_data),
            )))),
            Some(StorageBacking::U64) => Ok(TensorData::Numeric(Cow::Owned(into_vec_u8::<u64>(
                mem::take(&mut t.uint64_data),
            )))),
            Some(StorageBacking::I32) => Ok(TensorData::Numeric(Cow::Owned(into_vec_u8::<i32>(
                mem::take(&mut t.int32_data),
            )))),
            Some(StorageBacking::Strings) => {
                if t.string_data.is_empty() && self.shape.iter().any(|&d| d != 0) {
                    return Err(Error::MissingField("tensor data".to_string()));
                }
                Ok(TensorData::Strings(mem::take(&mut t.string_data)))
            }
            None => Err(Error::MissingField("tensor data".to_string())),
        }
    }
}

enum StorageBacking {
    F32,
    F64,
    I64,
    U64,
    I32,
    Strings,
}

fn storage_backing(dt: DataType) -> Option<StorageBacking> {
    match dt {
        DataType::Float | DataType::Complex64 => Some(StorageBacking::F32),
        DataType::Double | DataType::Complex128 => Some(StorageBacking::F64),
        DataType::Int64 => Some(StorageBacking::I64),
        DataType::Uint32 | DataType::Uint64 => Some(StorageBacking::U64),
        DataType::Int32
        | DataType::Int16
        | DataType::Int8
        | DataType::Int4
        | DataType::Int2
        | DataType::Uint16
        | DataType::Uint8
        | DataType::Uint4
        | DataType::Uint2
        | DataType::Bool
        | DataType::Float16
        | DataType::Bfloat16
        | DataType::Float8e4m3fn
        | DataType::Float8e4m3fnuz
        | DataType::Float8e5m2
        | DataType::Float8e5m2fnuz
        | DataType::Float8e8m0
        | DataType::Float4e2m1 => Some(StorageBacking::I32),
        DataType::String => Some(StorageBacking::Strings),
        DataType::Undefined => None,
    }
}

fn slice_bytes_as<T: Copy>(slice: &[T]) -> &[u8] {
    assert!(mem::size_of::<T>() > 0, "zero-sized types not supported");
    unsafe { slice::from_raw_parts(slice.as_ptr() as *const u8, mem::size_of_val(slice)) }
}

fn into_vec_u8<T: Copy>(v: Vec<T>) -> Vec<u8> {
    let t_size = mem::size_of::<T>();
    if t_size == 0 {
        return Vec::new();
    }
    let mut v = ManuallyDrop::new(v);
    let ptr = v.as_mut_ptr() as *mut u8;
    let len = v.len().checked_mul(t_size).expect("length overflow");
    let cap = v.capacity().checked_mul(t_size).expect("capacity overflow");
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}
