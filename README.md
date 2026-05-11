# onnx-extractor

A tiny and lightweight ONNX model parser for extracting tensor shapes, operations, and raw data with mmap and external data loading support. Every function in this crate provides zero-copy access to the underlying tensor data.

## Model Loading

```rust
use onnx_extractor::OnnxModel;

// Load from file (uses mmap)
let model = OnnxModel::load_from_file("model.onnx")?;

// Load from bytes
let bytes = std::fs::read("model.onnx")?;
let model = OnnxModel::load_from_bytes(bytes)?;
```

## Model Functions

```rust
// Basic info
model.print_summary();
model.print_model_info();

// Tensor access
let tensor = model.tensors().get("input_name"); // Returns Option<&OnnxTensor>
let tensor_names = model.tensors().keys(); // Iterator<Item = &String>
let inputs = model.get_input_tensors(); // Iterator<Item = &OnnxTensor>
let outputs = model.get_output_tensors(); // Iterator<Item = &OnnxTensor>
let weights = model.get_weight_tensors(); // Iterator<Item = &OnnxTensor>

// Extracting a tensor (moves out of model so data can outlive it)
let owned_tensor = model.take_tensor("weight"); // Option<OnnxTensor>

// Operation access
let operation = model.get_operation("op_name"); // Option<&OnnxOperation>
let conv_ops = model.get_operations_by_type("Conv"); // Iterator<Item = &OnnxOperation>
let op_types = model.operation_types(); // Vec<String>
let op_counts = model.count_operations_by_type(); // HashMap<&str, usize>

// Execution order
let topo_order = model.topological_order()?; // Result<Vec<&OnnxOperation>, Error>
let exec_order = model.execution_order()?; // Result<Vec<&OnnxOperation>, Error>
```

## Tensor Functions

```rust
let tensor = model.tensors().get("weight").unwrap();

// Shape and type info
println!("Name: {}", tensor.name());
println!("Shape: {:?}", tensor.shape());
println!("Data type: {:?}", tensor.data_type());

// Borrow tensor data
let data_ref = tensor.data()?; // Returns Result<TensorDataRef<'_>, Error>
println!("Data size: {} bytes", data_ref.len());

// Get data as contiguous byte slice
// Works for Raw and Numeric variants; returns Error for Strings as they are not contiguous in memory
let bytes: &[u8] = data_ref.as_slice()?;

// Access string elements for String tensors
if let Ok(strings) = data_ref.strings() {
    for s in strings {
        println!("String element: {:?}", s);
    }
}

// Access typed data
if let TensorDataRef::F32(floats) = data_ref {
    println!("First float: {}", floats[0]);
}

// Consume tensor and get owned data
// This allows the data to outlive the model itself
if let Some(owned_tensor) = model.take_tensor("weight") {
    let owned_data = owned_tensor.into_data()?; // Returns Result<TensorData, Error>
}
```

### Tensor Data Enums

The `data()` method returns a `TensorDataRef` which borrows from the model:

```rust
pub enum TensorDataRef<'a> {
    /// Contiguous buffer (mmap-backed or loaded)
    Raw(Bytes),
    /// String tensor elements, each Arc-backed
    Strings(&'a [Bytes]),
    /// Typed numeric data
    F32(&'a [f32]),
    F64(&'a [f64]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    U64(&'a [u64]),
}
```

The `into_data()` method returns an owned `TensorData` (using Bytes or Vec storage from the protobuf):

```rust
pub enum TensorData {
    /// Contiguous buffer (mmap-backed or loaded)
    Raw(Bytes),
    /// String tensor elements, each Arc-backed
    Strings(Vec<Bytes>),
    /// Typed numeric data
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
}
```

## Operation Functions

```rust
let op = model.get_operation("conv1").unwrap();

// Basic info
println!("Type: {}", op.op_type());
println!("Inputs: {:?}", op.inputs());
println!("Outputs: {:?}", op.outputs());

// Attribute access
let attributes = op.attributes();
if let Some(attr) = attributes.get("kernel_shape") {
    let kernel_size = attr.as_ints(); // Option<&[i64]>
}

let stride = attributes.get("stride").and_then(|a| a.as_int()); // Option<i64>
let activation = attributes.get("activation").and_then(|a| a.as_string()); // Option<&str>
```

## Data Types

Access the `DataType` enum for type checking:

```rust
use onnx_extractor::DataType;

let tensor = model.tensors().get("input").unwrap();
match tensor.data_type() {
    DataType::Float => println!("32-bit float"),
    DataType::Double => println!("64-bit float"),
    DataType::Int32 => println!("32-bit int"),
    _ => println!("Other type"),
}

// Type properties
let size = tensor.data_type().size_in_bytes(); // Option<usize>
let is_float = tensor.data_type().is_float();
let is_int = tensor.data_type().is_integer();
```

## External Data Support

ONNX models can store large tensor data in external files. This crate supports lazy loading of external data with automatic caching:

```rust
// Load model with external data files
let model = OnnxModel::load_from_file("large_model.onnx")?;

// External data files (e.g., "large_model.onnx.data") are automatically discovered
// and loaded lazily when tensor data is accessed

let tensor = model.tensors().get("large_weight").unwrap();

// Data is loaded from external file on first access and cached for subsequent use
let data = tensor.data()?;
println!("Loaded {} bytes from external file", data.len());

// Multiple tensors can share the same external file efficiently
// The file is only loaded once and cached
```

### External Data Features

- **Lazy Loading**: External files are only loaded when tensor data is accessed
- **Shared Caching**: Multiple tensors sharing the same external file benefit from caching
- **Offset & Length**: Supports reading specific ranges from large external files

## About the protobuf (`onnx.proto`)

This crate generates Rust types from the ONNX protobuf at build time using `prost-build`.

## Platform Notes

- Endianness: Tensor data follows the raw byte representation as defined in the ONNX specification.

## License

MIT
