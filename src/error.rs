use std::fmt;

/// Custom error type for onnx-extractor
#[derive(Debug)]
pub enum Error {
    /// I/O error when reading files
    Io(std::io::Error),
    /// Protobuf decoding error
    Decode(prost::DecodeError),
    /// UTF-8 conversion error
    Utf8(std::str::Utf8Error),
    /// Model structure error
    InvalidModel(String),
    /// Missing required field
    MissingField(String),
    /// Unsupported feature
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {}", e),
            Error::Decode(e) => write!(f, "Protobuf decode error: {}", e),
            Error::Utf8(e) => write!(f, "UTF-8 conversion error: {}", e),
            Error::InvalidModel(msg) => write!(f, "Invalid model: {}", msg),
            Error::MissingField(field) => write!(f, "Missing required field: {}", field),
            Error::Unsupported(feature) => write!(f, "Unsupported feature: {}", feature),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Decode(e) => Some(e),
            Error::Utf8(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<prost::DecodeError> for Error {
    fn from(err: prost::DecodeError) -> Self {
        Error::Decode(err)
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(err: std::str::Utf8Error) -> Self {
        Error::Utf8(err)
    }
}

impl From<std::num::ParseIntError> for Error {
    fn from(err: std::num::ParseIntError) -> Self {
        Error::InvalidModel(format!("Invalid integer in model metadata: {}", err))
    }
}
