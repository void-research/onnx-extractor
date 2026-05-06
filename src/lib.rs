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
//! `OnnxTensor::data()` borrows tensor data without copying the underlying bytes.
//! Raw and Strings variants increment Arc refcounts, Numeric borrows directly.
//!
//! `OnnxTensor::into_data()` returns owned data:
//! - Raw returns Arc-backed bytes
//! - Numeric performs zero-copy reinterpretation from typed fields
//! - Strings returns Arc-backed elements
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
pub use tensor::{OnnxTensor, TensorData};
pub use types::{AttributeValue, DataType};
