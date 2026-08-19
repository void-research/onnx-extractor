use prost::bytes::Bytes;

use crate::{Error, Graph, Tensor};

/// ONNX attribute values
#[derive(Debug)]
pub enum AttributeValue {
    Float(f32),
    Int(i64),
    String(Bytes),
    Tensor(Box<Tensor>),
    Graph(Box<Graph>),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<Bytes>),
    Tensors(Box<[Tensor]>),
    Graphs(Box<[Graph]>),
}

impl AttributeValue {
    /// Get string value as raw bytes without UTF-8 validation.
    pub fn as_string(&self) -> Option<&Bytes> {
        match self {
            AttributeValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get string value as a validated UTF-8 `&str`.
    ///
    /// Returns `Err` if the variant is not `String` or if the bytes are not valid UTF-8.
    /// The returned `&str` borrows directly from the underlying buffer with no copy.
    pub fn as_string_validated(&self) -> Result<&str, Error> {
        match self {
            AttributeValue::String(s) => Ok(std::str::from_utf8(s)?),
            _ => Err(Error::MissingField("string attribute")),
        }
    }

    /// Extract string value as owned `Bytes` without copying.
    pub fn into_string(self) -> Option<Bytes> {
        match self {
            AttributeValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get string array value as raw bytes without UTF-8 validation.
    pub fn as_strings(&self) -> Option<&[Bytes]> {
        match self {
            AttributeValue::Strings(s) => Some(s),
            _ => None,
        }
    }

    /// Get string array value as validated `&str` entries.
    ///
    /// Returns `Err` if the variant is not `Strings` or if any entry is not valid UTF-8.
    /// Each `&str` borrows directly from the underlying buffer with no copy,
    /// but the returned `Box<[&str]>` is a new allocation for the pointer array.
    pub fn as_strings_validated(&self) -> Result<Box<[&str]>, Error> {
        match self {
            AttributeValue::Strings(strings) => strings
                .iter()
                .map(|s| Ok(std::str::from_utf8(s)?))
                .collect(),
            _ => Err(Error::MissingField("strings attribute")),
        }
    }

    /// Extract string array value as owned `Vec<Bytes>` without copying.
    pub fn into_strings(self) -> Option<Vec<Bytes>> {
        match self {
            AttributeValue::Strings(s) => Some(s),
            _ => None,
        }
    }

    /// Get float value if the variant is `Float`.
    pub fn as_float(&self) -> Option<f32> {
        match self {
            AttributeValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Get integer value if the variant is `Int`.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            AttributeValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Borrow float slice if the variant is `Floats`.
    pub fn as_floats(&self) -> Option<&[f32]> {
        match self {
            AttributeValue::Floats(f) => Some(f),
            _ => None,
        }
    }

    /// Extract float vector as owned `Vec<f32>` without copying if the variant is `Floats`.
    pub fn into_floats(self) -> Option<Vec<f32>> {
        match self {
            AttributeValue::Floats(f) => Some(f),
            _ => None,
        }
    }

    /// Borrow integer slice if the variant is `Ints`.
    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            AttributeValue::Ints(i) => Some(i),
            _ => None,
        }
    }

    /// Extract integer vector as owned `Vec<i64>` without copying if the variant is `Ints`.
    pub fn into_ints(self) -> Option<Vec<i64>> {
        match self {
            AttributeValue::Ints(i) => Some(i),
            _ => None,
        }
    }

    /// Borrow tensor if the variant is `Tensor`.
    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            AttributeValue::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// Extract tensor as owned `Box<Tensor>` without copying if the variant is `Tensor`.
    pub fn into_tensor(self) -> Option<Box<Tensor>> {
        match self {
            AttributeValue::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// Borrow subgraph if the variant is `Graph`.
    pub fn as_graph(&self) -> Option<&Graph> {
        match self {
            AttributeValue::Graph(g) => Some(g),
            _ => None,
        }
    }

    /// Extract subgraph as owned `Box<Graph>` without copying if the variant is `Graph`.
    pub fn into_graph(self) -> Option<Box<Graph>> {
        match self {
            AttributeValue::Graph(g) => Some(g),
            _ => None,
        }
    }

    /// Borrow tensor slice if the variant is `Tensors`.
    pub fn as_tensors(&self) -> Option<&[Tensor]> {
        match self {
            AttributeValue::Tensors(t) => Some(t),
            _ => None,
        }
    }

    /// Extract tensor slice as owned `Box<[Tensor]>` without copying if the variant is `Tensors`.
    pub fn into_tensors(self) -> Option<Box<[Tensor]>> {
        match self {
            AttributeValue::Tensors(t) => Some(t),
            _ => None,
        }
    }

    /// Borrow subgraph slice if the variant is `Graphs`.
    pub fn as_graphs(&self) -> Option<&[Graph]> {
        match self {
            AttributeValue::Graphs(g) => Some(g),
            _ => None,
        }
    }

    /// Extract subgraph slice as owned `Box<[Graph]>` without copying if the variant is `Graphs`.
    pub fn into_graphs(self) -> Option<Box<[Graph]>> {
        match self {
            AttributeValue::Graphs(g) => Some(g),
            _ => None,
        }
    }
}
