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
    /// Get string value as a validated UTF-8 `&str`.
    ///
    /// Returns `Err` if the variant is not `String` or if the bytes are not valid UTF-8.
    /// The returned `&str` borrows directly from the underlying buffer with no copy.
    pub fn as_string(&self) -> Result<&str, Error> {
        match self {
            AttributeValue::String(s) => Ok(std::str::from_utf8(s)?),
            _ => Err(Error::MissingField("string attribute".to_string())),
        }
    }

    /// Get string array value as validated `&str` entries.
    ///
    /// Returns `Err` if the variant is not `Strings` or if any entry is not valid UTF-8.
    /// Each `&str` borrows directly from the underlying buffer with no copy,
    /// but the returned `Box<[&str]>` is a new allocation for the pointer array.
    pub fn as_strings(&self) -> Result<Box<[&str]>, Error> {
        match self {
            AttributeValue::Strings(strings) => strings
                .iter()
                .map(|s| Ok(std::str::from_utf8(s)?))
                .collect(),
            _ => Err(Error::MissingField("strings attribute".to_string())),
        }
    }
}
