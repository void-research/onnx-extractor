//! # onnx-extractor
//!
//! A lightweight ONNX model parser for extracting tensor shapes, operations, and data.
//!
//! This crate provides a simple interface to parse ONNX models and extract:
//! - Tensor information (shapes, data types, raw data)
//! - Operation details (inputs, outputs, attributes)
//! - Model structure (inputs, outputs, graph topology)
//!
//! ## Zero-Copy Design
//!
//! `OnnxTensor::data()` returns a `TensorDataRef` which borrows tensor data without copying:
//! - Raw variants use shared ownership (`Bytes`)
//! - Numeric variants borrow directly (`&[T]`)
//! - Strings variants borrow elements (`&[Bytes]`)
//!
//! `OnnxTensor::into_data()` returns owned `TensorData` without copying:
//! - Raw returns shared ownership (`Bytes`)
//! - Numeric returns owned vectors (`Vec<T>`)
//! - Strings returns owned vectors of shared elements (`Vec<Bytes>`)
//!
//! Endianness: Multi-byte interpretations assume little-endian platforms.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use onnx_extractor::OnnxModel;
//!
//! let model = OnnxModel::load_from_file("model.onnx")?;
//! model.print_model_info();
//!
//! // Access tensor information
//! if let Some(tensor) = model.tensors().get("input") {
//!     println!("Input shape: {:?}", tensor.shape());
//!     let data = tensor.data()?;
//!     if let Ok(bytes) = data.as_slice() {
//!         println!("Data size: {} bytes", bytes.len());
//!     }
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// include generated protobuf code inside a small module so we can silence
// lints and doc warnings originating from the generated file only.
#[allow(clippy::all)]
#[allow(rustdoc::all)]
mod onnx_generated {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub(crate) mod proto_adapter;
pub(crate) use onnx_generated::*;

pub mod error;
pub mod external_data;
pub mod model;
pub mod operation;
pub mod tensor;
pub mod types;

pub use error::Error;
pub use model::OnnxModel;
pub use operation::OnnxOperation;
pub use prost::bytes::Bytes;
pub use tensor::{OnnxTensor, TensorData, TensorDataRef};
pub use types::{AttributeValue, DataType};
