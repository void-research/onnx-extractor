//! # onnx-extractor
//!
//! A minimal ONNX model loader for extracting weights, tensors, operations, and graph structure.
//!
//! This crate provides a simple interface to load ONNX models and extract:
//! - Tensor weights & raw data (zero-copy slices, owned buffers, external files)
//! - Tensor shapes, dimensions, and data types
//! - Operations and attributes (nodes, inputs/outputs, subgraphs)
//! - Graph topology and execution order
//!
//! ## Zero-Copy Design
//!
//! `Tensor::data()` returns a `TensorDataRef` which borrows tensor data without copying:
//! - Raw variants use shared ownership (`Bytes`)
//! - Numeric variants borrow directly (`&[T]`)
//! - Strings variants borrow elements (`&[Bytes]`)
//!
//! `Tensor::into_data()` returns owned `TensorData` without copying:
//! - Raw returns shared ownership (`Bytes`)
//! - Numeric returns owned vectors (`Vec<T>`)
//! - Strings returns owned vectors of shared elements (`Vec<Bytes>`)
//!
//! Endianness: Tensor data follows the raw byte representation as defined in the ONNX specification.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use onnx_extractor::Model;
//!
//! let model = Model::load_from_file("model.onnx")?;
//! println!("{}", model);
//!
//! // Access tensor shape and data
//! if let Some(tensor) = model.graph().tensors().get("weight_1") {
//!     println!("Weight shape: {:?}", tensor.shape());
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

pub mod attribute_value;
pub mod data_type;
pub mod error;
mod external_data;
pub mod graph;
pub mod model;
pub mod operation;
pub mod tensor;

pub use attribute_value::AttributeValue;
pub use data_type::DataType;
pub use error::Error;
pub use graph::Graph;
pub use model::Model;
pub use operation::Operation;
pub use prost::bytes::Bytes;
pub use tensor::{Tensor, TensorData, TensorDataRef};
