/// Custom error type for onnx-extractor
#[derive(Debug)]
pub enum Error {
    /// I/O error when reading files
    Io(std::io::Error),
    /// Protobuf decoding error
    Decode(prost::DecodeError),
    /// UTF-8 conversion error
    Utf8(std::str::Utf8Error),
    /// Integer parse error
    ParseInt(std::num::ParseIntError),
    /// Lock poisoned error when accessing external data cache
    ExternalDataLockPoisoned,
    /// Invalid graph structure
    InvalidGraph,
    /// Missing required field
    MissingField(&'static str),
    /// Model contains external tensor data, but was loaded without a filesystem path
    ExternalDataRequiresPath,
    /// Unsupported attribute type
    UnsupportedAttributeType(i32),
    /// External data slice range exceeds file size
    ExternalDataOutOfBounds {
        start: u64,
        end: u64,
        file_size: u64,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Decode(e) => write!(f, "Protobuf decode error: {e}"),
            Error::Utf8(e) => write!(f, "UTF-8 conversion error: {e}"),
            Error::ParseInt(e) => write!(f, "Parse int error: {e}"),
            Error::ExternalDataLockPoisoned => write!(f, "Lock poisoned"),
            Error::InvalidGraph => write!(
                f,
                "Invalid graph: graph has cycles or unresolved dependencies"
            ),
            Error::MissingField(field) => write!(f, "Missing required field: {field}"),
            Error::ExternalDataRequiresPath => write!(
                f,
                "Model contains external tensor data, but was loaded without a filesystem path"
            ),
            Error::UnsupportedAttributeType(t) => write!(f, "Unsupported attribute type: {t}"),
            Error::ExternalDataOutOfBounds {
                start,
                end,
                file_size,
            } => write!(
                f,
                "External data range {start}..{end} exceeds file size {file_size}"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Decode(e) => Some(e),
            Error::Utf8(e) => Some(e),
            Error::ParseInt(e) => Some(e),
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

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        Error::ExternalDataLockPoisoned
    }
}

impl From<std::num::ParseIntError> for Error {
    fn from(err: std::num::ParseIntError) -> Self {
        Error::ParseInt(err)
    }
}
