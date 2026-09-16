use crate::attribute_proto::AttributeType;
use crate::external_data::{ExternalDataInfo, ExternalDataLoader};
use crate::tensor::TensorDataLocation;
use crate::{
    AttributeProto, AttributeValue, DataType, Error, Graph, GraphProto, NodeProto, Operation,
    Tensor, TensorProto, TypeProto, tensor_shape_proto::dimension::Value, type_proto,
};
use std::collections::hash_map::{Entry, HashMap};
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
        let loader = external_data_loader.ok_or(Error::ExternalDataRequiresPath)?;
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

    Ok(Tensor::new(tensor.name, tensor.dims, data_type, data))
}

fn tensor_from_value_info(name: Option<String>, vi_type: Option<TypeProto>) -> Option<Tensor> {
    let Some(type_proto::Value::TensorType(tensor_type)) = vi_type.and_then(|t| t.value) else {
        return None;
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

    let data_type = DataType::from_onnx_type(tensor_type.elem_type.unwrap_or(0));

    Some(Tensor::new(
        name,
        shape,
        data_type,
        TensorDataLocation::None,
    ))
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
        let name = onnx_tensor
            .name()
            .filter(|n| !n.is_empty())
            .ok_or(Error::MissingField("initialiser tensor name"))?;
        tensors.insert(name.to_string(), onnx_tensor);
    }

    // Parse input tensor info and extract input names
    let mut inputs = Vec::with_capacity(graph.input.len());
    for input in graph.input {
        let name = input
            .name
            .filter(|n| !n.is_empty())
            .ok_or(Error::MissingField("graph input name"))?;
        // If the name is already in tensors, it's an initialiser, so we skip adding it to inputs/tensors
        if let Entry::Vacant(entry) = tensors.entry(name) {
            inputs.push(entry.key().clone());
            if let Some(tensor) = tensor_from_value_info(Some(entry.key().clone()), input.r#type) {
                entry.insert(tensor);
            }
        }
    }

    // Parse output tensor info and extract output names
    let mut outputs = Vec::with_capacity(graph.output.len());
    for output in graph.output {
        let name = output
            .name
            .filter(|n| !n.is_empty())
            .ok_or(Error::MissingField("graph output name"))?;
        outputs.push(name.clone());
        if let Entry::Vacant(entry) = tensors.entry(name)
            && let Some(tensor) = tensor_from_value_info(Some(entry.key().clone()), output.r#type)
        {
            entry.insert(tensor);
        }
    }

    // Parse value_info for intermediate tensor shapes and types
    for value_info in graph.value_info {
        if let Some(name) = value_info.name.filter(|n| !n.is_empty())
            && let Entry::Vacant(entry) = tensors.entry(name)
            && let Some(tensor) =
                tensor_from_value_info(Some(entry.key().clone()), value_info.r#type)
        {
            entry.insert(tensor);
        }
    }

    // Parse operations/nodes
    let operations = graph
        .node
        .into_iter()
        .map(|node| operation_from_node_proto(node, external_data_loader))
        .collect::<Result<Vec<_>, Error>>()?;

    Ok(Graph::new(graph.name, tensors, operations, inputs, outputs))
}

/// Create Operation from ONNX NodeProto
pub(crate) fn operation_from_node_proto(
    node: NodeProto,
    external_data_loader: Option<&Arc<ExternalDataLoader>>,
) -> Result<Operation, Error> {
    let op_type = node
        .op_type
        .filter(|s| !s.is_empty())
        .ok_or(Error::MissingField("node op_type"))?;

    let attributes: HashMap<String, AttributeValue> = node
        .attribute
        .into_iter()
        .map(|attr| parse_attribute_proto(attr, external_data_loader))
        .collect::<Result<HashMap<_, _>, Error>>()?;

    Ok(Operation::new(
        node.name,
        op_type,
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
    let name = attr
        .name
        .filter(|n| !n.is_empty())
        .ok_or(Error::MissingField("attribute name"))?;

    let raw_type = attr.r#type.ok_or(Error::MissingField("attribute type"))?;

    let attr_type =
        AttributeType::try_from(raw_type).map_err(|_| Error::UnsupportedAttributeType(raw_type))?;

    let value = match attr_type {
        AttributeType::Float => AttributeValue::Float(attr.f.unwrap_or(0.0)),
        AttributeType::Int => AttributeValue::Int(attr.i.unwrap_or(0)),
        AttributeType::String => AttributeValue::String(attr.s.unwrap_or_default()),
        AttributeType::Tensor => {
            let tensor = attr.t.ok_or(Error::MissingField("tensor attribute data"))?;
            let onnx_tensor = tensor_from_proto(tensor, external_data_loader)?;
            AttributeValue::Tensor(Box::new(onnx_tensor))
        }
        AttributeType::Graph => {
            let graph = attr.g.ok_or(Error::MissingField("graph attribute data"))?;
            let onnx_graph = graph_from_proto(graph, external_data_loader)?;
            AttributeValue::Graph(Box::new(onnx_graph))
        }
        AttributeType::Floats => AttributeValue::Floats(attr.floats),
        AttributeType::Ints => AttributeValue::Ints(attr.ints),
        AttributeType::Strings => AttributeValue::Strings(attr.strings),
        AttributeType::Tensors => AttributeValue::Tensors(
            attr.tensors
                .into_iter()
                .map(|tensor| tensor_from_proto(tensor, external_data_loader))
                .collect::<Result<Box<[Tensor]>, Error>>()?,
        ),
        AttributeType::Graphs => AttributeValue::Graphs(
            attr.graphs
                .into_iter()
                .map(|graph| graph_from_proto(graph, external_data_loader))
                .collect::<Result<Box<[Graph]>, Error>>()?,
        ),
        AttributeType::Undefined
        | AttributeType::SparseTensor
        | AttributeType::SparseTensors
        | AttributeType::TypeProto
        | AttributeType::TypeProtos => return Err(Error::UnsupportedAttributeType(raw_type)),
    };

    Ok((name, value))
}
