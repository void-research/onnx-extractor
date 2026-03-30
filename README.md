# onnx-extractor

A tiny and lightweight ONNX model parser for extracting tensor shapes, operations, and raw data with zero-copy, mmap, and external data loading support.

## Model Loading

```rust
use onnx_extractor::OnnxModel;

// Load from file
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
let tensor = model.get_tensor("input_name");
let tensor_names = model.tensor_names();
let inputs = model.get_input_tensors();
let outputs = model.get_output_tensors();
let weights = model.get_weight_tensors();

// Operation access
let operation = model.get_operation("op_name");
let conv_ops = model.get_operations_by_type("Conv");
let op_types = model.operation_types();
let op_counts = model.count_operations_by_type();

// Execution order
let topo_order = model.topological_order()?;
let exec_order = model.execution_order()?;
```

## Tensor Functions

```rust
let tensor = model.get_tensor("weight").unwrap();

// Shape and type info
println!("Name: {}", tensor.name());
println!("Shape: {:?}", tensor.shape());
println!("Data type: {:?}", tensor.data_type());

// Borrow tensor data
let tensor_data = tensor.data()?;
println!("Data size: {} bytes", tensor_data.len());

// Get data as byte slice
let bytes: Cow<'_, [u8]> = tensor_data.as_slice();

// Consume tensor and get owned data zero-copy
let owned_data = tensor.into_data()?;

// Copy/interpret data as typed buffer (little-endian)
let as_f32: Box<[f32]> = tensor.copy_data_as::<f32>()?;
let as_f64: Box<[f64]> = tensor.copy_data_as::<f64>()?;
let as_i32: Box<[i32]> = tensor.copy_data_as::<i32>()?;
let as_u8: Box<[u8]> = tensor.copy_data_as::<u8>()?;
```

### TensorData Variants

The `data()` and `into_data()` methods return a `TensorData` enum:

```rust
pub enum TensorData<'a> {
    /// Contiguous buffer from raw_data field, Arc-backed
    Raw(Bytes),
    /// Reinterpreted numeric data from typed fields
    Numeric(Cow<'a, [u8]>),
    /// String tensor elements, each Arc-backed
    Strings(Vec<Bytes>),
}
```

## Operation Functions

```rust
let op = model.get_operation("conv1").unwrap();

// Basic info
println!("Type: {}", op.op_type);
println!("Inputs: {:?}", op.inputs);
println!("Outputs: {:?}", op.outputs);

// Attribute access
let kernel_size = op.get_ints_attribute("kernel_shape");
let stride = op.get_int_attribute("stride");
let activation = op.get_string_attribute("activation");
let weight = op.get_float_attribute("alpha");

// Properties
let input_count = op.input_count();
let output_count = op.output_count();
let is_conv = op.is_op_type("Conv");
let has_bias = op.has_attribute("bias");
let attr_names = op.attribute_names();
```

## Data Types

Access the `DataType` enum for type checking:

```rust
use onnx_extractor::DataType;

let tensor = model.get_tensor("input").unwrap();
match tensor.data_type {
    DataType::Float => println!("32-bit float"),
    DataType::Double => println!("64-bit float"),
    DataType::Int32 => println!("32-bit int"),
    _ => println!("Other type"),
}

// Type properties
let size = tensor.data_type.size_in_bytes();
let is_float = tensor.data_type.is_float();
let is_int = tensor.data_type.is_integer();
```

## External Data Support

ONNX models can store large tensor data in external files. This crate supports lazy loading of external data with automatic caching:

```rust
// Load model with external data files
let model = OnnxModel::load_from_file("large_model.onnx")?;

// External data files (e.g., "large_model.onnx.data") are automatically discovered
// and loaded lazily when tensor data is accessed

let tensor = model.get_tensor("large_weight").unwrap();

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
- **Zero-Copy**: External data is stored as `Bytes` (Arc-backed) for cheap cloning

## About the protobuf (`onnx.proto`)

This crate generates Rust types from the ONNX protobuf at build time using `prost-build`.

## Platform Notes

- Byte and typed views assume little-endian platforms
- Raw tensor data follows the ONNX specification (IEEE 754 for floats, little-endian integers)

## License

MIT
