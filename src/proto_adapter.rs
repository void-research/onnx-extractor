use crate::external_data::{ExternalDataInfo, ExternalDataLoader};
use crate::tensor::TensorDataLocation;
use crate::{
    AttributeProto, AttributeValue, DataType, Error, Graph, GraphProto, NodeProto, Operation,
    Tensor, TensorProto, ValueInfoProto, tensor_shape_proto::dimension::Value, type_proto,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Create Tensor from ONNX TensorProto
pub(crate) fn tensor_from_proto(
    tensor: TensorProto,
    external_data_loader: Option<&Arc<ExternalDataLoader>>,
) -> Result<Tensor, Error> {
    let data_type = DataType::from_onnx_type(tensor.data_type.unwrap_or(0));

    // Determine data location (internal vs external vs mmap-backed raw)
    let data = if !tensor.external_data.is_empty() {
        // Tensor has external data
        let loader = external_data_loader.ok_or_else(|| {
            Error::InvalidModel(
                "Tensor has external data but no external data loader was provided".to_string(),
            )
        })?;
        let external_info =
            ExternalDataInfo::from_key_value_pairs(tensor.external_data, loader.clone())?;
        TensorDataLocation::External(external_info)
    } else if let Some(raw) = tensor.raw_data {
        // Keep raw_data as a Bytes reference (mmap-backed when loaded from file)
        TensorDataLocation::Mmap(raw)
    } else {
        match data_type {
            DataType::Undefined => TensorDataLocation::None,
            DataType::String => TensorDataLocation::MmapStrings(tensor.string_data),
            DataType::Float | DataType::Complex64 => TensorDataLocation::F32(tensor.float_data),
            DataType::Double | DataType::Complex128 => TensorDataLocation::F64(tensor.double_data),
            DataType::Int64 => TensorDataLocation::I64(tensor.int64_data),
            DataType::Uint32 | DataType::Uint64 => TensorDataLocation::U64(tensor.uint64_data),
            DataType::Int32
            | DataType::Int16
            | DataType::Int8
            | DataType::Int4
            | DataType::Int2
            | DataType::Uint16
            | DataType::Uint8
            | DataType::Uint4
            | DataType::Uint2
            | DataType::Bool
            | DataType::Float16
            | DataType::Bfloat16
            | DataType::Float8e4m3fn
            | DataType::Float8e4m3fnuz
            | DataType::Float8e5m2
            | DataType::Float8e5m2fnuz
            | DataType::Float8e8m0
            | DataType::Float4e2m1 => TensorDataLocation::I32(tensor.int32_data),
        }
    };

    Ok(Tensor::new(
        tensor.name.unwrap_or_default(),
        tensor.dims,
        data_type,
        data,
    ))
}

fn tensor_from_value_info(vi: ValueInfoProto) -> Result<Option<Tensor>, Error> {
    let Some(type_proto::Value::TensorType(tensor_type)) = vi.r#type.and_then(|t| t.value) else {
        return Ok(None);
    };

    let shape = tensor_type
        .shape
        .iter()
        .flat_map(|s| &s.dim)
        .map(|d| match d.value {
            Some(Value::DimValue(v)) => v,
            _ => -1,
        })
        .collect();

    let data_type = match tensor_type.elem_type {
        Some(0) => {
            return Err(Error::InvalidModel(
                "tensor elem_type must not be UNDEFINED (0)".to_string(),
            ));
        }
        Some(t) => DataType::from_onnx_type(t),
        None => return Err(Error::MissingField("tensor elem_type".to_string())),
    };

    Ok(Some(Tensor::new(
        vi.name.unwrap_or_default(),
        shape,
        data_type,
        TensorDataLocation::None,
    )))
}

/// Create Graph from ONNX GraphProto
pub(crate) fn graph_from_proto(
    graph: GraphProto,
    external_data_loader: Option<&Arc<ExternalDataLoader>>,
) -> Result<Graph, Error> {
    let mut tensors = HashMap::with_capacity(
        graph.initializer.len() + graph.value_info.len() + graph.input.len() + graph.output.len(),
    );

    // Parse initialiser tensors (weights/constants)
    for tensor in graph.initializer {
        let onnx_tensor = tensor_from_proto(tensor, external_data_loader)?;
        tensors.insert(onnx_tensor.name().to_string(), onnx_tensor);
    }

    // Parse input tensor info and extract input names
    let mut inputs = Vec::with_capacity(graph.input.len());
    for input in graph.input {
        let name = input.name.as_deref().unwrap_or_default();
        // If the name is already in tensors, it's an initialiser, so we skip adding it to inputs/tensors
        if !tensors.contains_key(name) {
            inputs.push(name.to_string());
            if let Some(tensor) = tensor_from_value_info(input)? {
                tensors.insert(tensor.name().to_string(), tensor);
            }
        }
    }

    // Parse value_info for intermediate tensor shapes and types
    for value_info in graph.value_info {
        if !tensors.contains_key(value_info.name.as_deref().unwrap_or_default())
            && let Some(tensor) = tensor_from_value_info(value_info)?
        {
            tensors.insert(tensor.name().to_string(), tensor);
        }
    }

    // Parse output tensor info and extract output names
    let mut outputs = Vec::with_capacity(graph.output.len());
    for output in graph.output {
        let name = output.name.as_deref().unwrap_or_default();
        outputs.push(name.to_string());
        if !tensors.contains_key(name)
            && let Some(tensor) = tensor_from_value_info(output)?
        {
            tensors.insert(tensor.name().to_string(), tensor);
        }
    }

    // Parse operations/nodes
    let operations = graph
        .node
        .into_iter()
        .map(|node| operation_from_node_proto(node, external_data_loader))
        .collect::<Result<Vec<_>, Error>>()?;

    Ok(Graph::new(
        graph.name.unwrap_or_default(),
        tensors,
        operations,
        inputs,
        outputs,
    ))
}

/// Create Operation from ONNX NodeProto
pub(crate) fn operation_from_node_proto(
    node: NodeProto,
    external_data_loader: Option<&Arc<ExternalDataLoader>>,
) -> Result<Operation, Error> {
    let attributes: HashMap<String, AttributeValue> = node
        .attribute
        .into_iter()
        .map(|attr| parse_attribute_proto(attr, external_data_loader))
        .collect::<Result<HashMap<_, _>, Error>>()?;

    Ok(Operation::new(
        node.name.unwrap_or_default(),
        node.op_type.unwrap_or_default(),
        node.input,
        node.output,
        attributes,
    ))
}

/// Parse ONNX attribute into a (name, AttributeValue) pair
///
/// Strings and string arrays are stored as `prost::bytes::Bytes` to avoid
/// mandatory UTF-8 validation during parsing. This allows zero-copy moves from
/// the protobuf structure.
pub(crate) fn parse_attribute_proto(
    attr: AttributeProto,
    external_data_loader: Option<&Arc<ExternalDataLoader>>,
) -> Result<(String, AttributeValue), Error> {
    let value = match attr.r#type.unwrap_or(0) {
        1 => Ok(AttributeValue::Float(attr.f.unwrap_or(0.0))),
        2 => Ok(AttributeValue::Int(attr.i.unwrap_or(0))),
        3 => Ok(AttributeValue::String(attr.s.unwrap_or_default())),
        4 => {
            let tensor = attr
                .t
                .ok_or_else(|| Error::MissingField("tensor attribute data".to_string()))?;
            let onnx_tensor = tensor_from_proto(tensor, external_data_loader)?;
            Ok(AttributeValue::Tensor(Box::new(onnx_tensor)))
        }
        5 => {
            let graph = attr
                .g
                .ok_or_else(|| Error::MissingField("graph attribute data".to_string()))?;
            let onnx_graph = graph_from_proto(graph, external_data_loader)?;
            Ok(AttributeValue::Graph(Box::new(onnx_graph)))
        }
        6 => Ok(AttributeValue::Floats(attr.floats)),
        7 => Ok(AttributeValue::Ints(attr.ints)),
        8 => Ok(AttributeValue::Strings(attr.strings)),
        9 => Ok(AttributeValue::Tensors(
            attr.tensors
                .into_iter()
                .map(|tensor| tensor_from_proto(tensor, external_data_loader))
                .collect::<Result<Box<[Tensor]>, Error>>()?,
        )),
        10 => Ok(AttributeValue::Graphs(
            attr.graphs
                .into_iter()
                .map(|graph| graph_from_proto(graph, external_data_loader))
                .collect::<Result<Box<[Graph]>, Error>>()?,
        )),
        n => Err(Error::Unsupported(format!("attribute type: {n}"))),
    }?;

    Ok((attr.name.unwrap_or_default(), value))
}
