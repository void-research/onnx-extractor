use prost::bytes::Bytes;
use std::{fmt, mem, slice};

use crate::{DataType, Error, external_data::ExternalDataInfo};

pub(crate) enum TensorDataLocation {
    None,
    External(Box<ExternalDataInfo>),
    Mmap(Bytes),
    MmapStrings(Vec<Bytes>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    I32(Vec<i32>),
}

impl fmt::Debug for TensorDataLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorDataLocation::None => f.write_str("None"),
            TensorDataLocation::External(info) => f.debug_tuple("External").field(info).finish(),
            TensorDataLocation::Mmap(b) => write!(f, "Mmap({} bytes)", b.len()),
            TensorDataLocation::MmapStrings(s) => write!(f, "MmapStrings({} elements)", s.len()),
            TensorDataLocation::F32(v) => write!(f, "F32({} elements)", v.len()),
            TensorDataLocation::F64(v) => write!(f, "F64({} elements)", v.len()),
            TensorDataLocation::I64(v) => write!(f, "I64({} elements)", v.len()),
            TensorDataLocation::U64(v) => write!(f, "U64({} elements)", v.len()),
            TensorDataLocation::I32(v) => write!(f, "I32({} elements)", v.len()),
        }
    }
}

/// Zero-copy tensor data reference
pub enum TensorDataRef<'a> {
    Raw(Bytes),
    Strings(&'a [Bytes]),
    F32(&'a [f32]),
    F64(&'a [f64]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    U64(&'a [u64]),
}

impl<'a> TensorDataRef<'a> {
    /// Total byte length across all variants
    ///
    /// For `Strings`, returns the sum of all string element byte lengths.
    /// If all `Strings` are empty, returns 0.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            TensorDataRef::Raw(b) => b.len(),
            TensorDataRef::Strings(parts) => parts.iter().map(Bytes::len).sum(),
            TensorDataRef::F32(v) => mem::size_of_val(*v),
            TensorDataRef::F64(v) => mem::size_of_val(*v),
            TensorDataRef::I32(v) => mem::size_of_val(*v),
            TensorDataRef::I64(v) => mem::size_of_val(*v),
            TensorDataRef::U64(v) => mem::size_of_val(*v),
        }
    }

    /// Returns true if data contains no elements
    ///
    /// For `Raw` and numeric variants, equivalent to `len() == 0`.
    /// For `Strings`, checks if the slice of string elements is empty.
    #[must_use]
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
    /// Raw and numeric variants borrow directly.
    /// Returns `None` for the `Strings` variant as string arrays are not contiguous byte buffers.
    #[must_use]
    pub fn as_slice(&self) -> Option<&[u8]> {
        match self {
            TensorDataRef::Raw(b) => Some(b),
            TensorDataRef::F32(v) => Some(slice_as_u8(v)),
            TensorDataRef::F64(v) => Some(slice_as_u8(v)),
            TensorDataRef::I32(v) => Some(slice_as_u8(v)),
            TensorDataRef::I64(v) => Some(slice_as_u8(v)),
            TensorDataRef::U64(v) => Some(slice_as_u8(v)),
            TensorDataRef::Strings(_) => None,
        }
    }

    /// Access the string elements if the variant is `Strings`. Returns `None` otherwise.
    #[must_use]
    pub fn strings(&self) -> Option<&'a [Bytes]> {
        match self {
            TensorDataRef::Strings(v) => Some(v),
            _ => None,
        }
    }
}

impl fmt::Debug for TensorDataRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorDataRef::Raw(b) => write!(f, "Raw({} bytes)", b.len()),
            TensorDataRef::Strings(s) => write!(f, "Strings({} elements)", s.len()),
            TensorDataRef::F32(v) => write!(f, "F32({} elements)", v.len()),
            TensorDataRef::F64(v) => write!(f, "F64({} elements)", v.len()),
            TensorDataRef::I32(v) => write!(f, "I32({} elements)", v.len()),
            TensorDataRef::I64(v) => write!(f, "I64({} elements)", v.len()),
            TensorDataRef::U64(v) => write!(f, "U64({} elements)", v.len()),
        }
    }
}

/// Zero-copy owned tensor data
pub enum TensorData {
    Raw(Bytes),
    Strings(Vec<Bytes>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
}

impl TensorData {
    /// Total byte length across all variants
    ///
    /// For `Strings`, returns the sum of all string element byte lengths.
    /// If all `Strings` are empty, returns 0.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            TensorData::Raw(b) => b.len(),
            TensorData::Strings(parts) => parts.iter().map(Bytes::len).sum(),
            TensorData::F32(v) => mem::size_of_val(v.as_slice()),
            TensorData::F64(v) => mem::size_of_val(v.as_slice()),
            TensorData::I32(v) => mem::size_of_val(v.as_slice()),
            TensorData::I64(v) => mem::size_of_val(v.as_slice()),
            TensorData::U64(v) => mem::size_of_val(v.as_slice()),
        }
    }

    /// Returns true if data contains no elements
    ///
    /// For `Raw` and numeric variants, equivalent to `len() == 0`.
    /// For `Strings`, checks if the vector of string elements is empty.
    #[must_use]
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
    /// Raw and numeric variants borrow directly.
    /// Returns `None` for the `Strings` variant as string arrays are not contiguous byte buffers.
    #[must_use]
    pub fn as_slice(&self) -> Option<&[u8]> {
        match self {
            TensorData::Raw(b) => Some(b),
            TensorData::F32(v) => Some(slice_as_u8(v)),
            TensorData::F64(v) => Some(slice_as_u8(v)),
            TensorData::I32(v) => Some(slice_as_u8(v)),
            TensorData::I64(v) => Some(slice_as_u8(v)),
            TensorData::U64(v) => Some(slice_as_u8(v)),
            TensorData::Strings(_) => None,
        }
    }

    /// Access the string elements if the variant is `Strings`. Returns `None` otherwise.
    #[must_use]
    pub fn strings(&self) -> Option<&[Bytes]> {
        match self {
            TensorData::Strings(v) => Some(v),
            _ => None,
        }
    }
}

impl fmt::Debug for TensorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorData::Raw(b) => write!(f, "Raw({} bytes)", b.len()),
            TensorData::Strings(s) => write!(f, "Strings({} elements)", s.len()),
            TensorData::F32(v) => write!(f, "F32({} elements)", v.len()),
            TensorData::F64(v) => write!(f, "F64({} elements)", v.len()),
            TensorData::I32(v) => write!(f, "I32({} elements)", v.len()),
            TensorData::I64(v) => write!(f, "I64({} elements)", v.len()),
            TensorData::U64(v) => write!(f, "U64({} elements)", v.len()),
        }
    }
}

/// An ONNX tensor with a name, shape, data type, and optional underlying data
#[derive(Debug)]
pub struct Tensor {
    name: Option<String>,
    shape: Vec<i64>,
    data_type: DataType,
    data: TensorDataLocation,
}

impl Tensor {
    pub(crate) fn new(
        name: Option<String>,
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

    /// Tensor name
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Tensor shape dimensions
    #[must_use]
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Tensor data type
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns true if this tensor contains data.
    ///
    /// This check does not trigger loading or memory-mapping of external files.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !matches!(self.data, TensorDataLocation::None)
    }

    /// Borrow tensor data
    ///
    /// - All variants are returned without copying the underlying tensor data.
    /// - External data is loaded from disk if not already in memory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingField`] if the tensor has no data, or [`Error::Io`] /
    /// [`Error::ExternalDataOutOfBounds`] / [`Error::ExternalDataLockPoisoned`] if loading
    /// external data fails.
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
            TensorDataLocation::None => Err(Error::MissingField("tensor data")),
        }
    }

    /// Consume tensor and return owned data
    ///
    /// - All variants are returned without copying the underlying tensor data.
    /// - External data is loaded from disk if not already in memory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingField`] if the tensor has no data, or [`Error::Io`] /
    /// [`Error::ExternalDataOutOfBounds`] / [`Error::ExternalDataLockPoisoned`] if loading
    /// external data fails.
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
            TensorDataLocation::None => Err(Error::MissingField("tensor data")),
        }
    }
}

fn slice_as_u8<T>(slice: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(slice.as_ptr().cast::<u8>(), mem::size_of_val(slice)) }
}
