use crate::external_data::{ExternalDataInfo, ExternalDataLoader};
use crate::tensor::TensorDataLocation;
use crate::{
    AttributeProto, AttributeValue, DataType, Error, Graph, NodeProto, Operation, Tensor,
    TensorProto,
};
use std::{collections::HashMap, sync::Arc};

/// Centralised adapter functions that translate generated protobuf types into
/// crate-native types. Keep all direct proto-field usage here so future changes
/// to `onnx.proto` need only update this file.
///
/// Zero-copy policy: we prefer moving/borrowing from the generated proto
/// structures. We avoid cloning where `prost` provides owned fields.
///
/// Create Tensor from ONNX TensorProto
pub(crate) fn tensor_from_proto(
    tensor: TensorProto,
    external_data_loader: &Option<Arc<ExternalDataLoader>>,
) -> Result<Tensor, Error> {
    let data_type = DataType::from_onnx_type(tensor.data_type.unwrap_or(0));

    // Determine data location (internal vs external vs mmap-backed raw)
    let data = if !tensor.external_data.is_empty() {
        // Tensor has external data
        if let Some(loader) = external_data_loader {
            let external_info =
                ExternalDataInfo::from_key_value_pairs(&tensor.external_data, loader.clone())?;
            TensorDataLocation::External(external_info)
        } else {
            return Err(Error::InvalidModel(
                "Tensor has external data but no external data loader was provided".to_string(),
            ));
        }
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

/// Create Operation from ONNX NodeProto
pub(crate) fn operation_from_node_proto(
    node: NodeProto,
    external_data_loader: &Option<Arc<ExternalDataLoader>>,
) -> Result<Operation, Error> {
    let attributes: HashMap<String, AttributeValue> = node
        .attribute
        .into_iter()
        .map(|mut attr| {
            let attr_name = attr.name.take().unwrap_or_default();
            parse_attribute_proto(attr, external_data_loader).map(|v| (attr_name, v))
        })
        .collect::<Result<HashMap<_, _>, Error>>()?;

    Ok(Operation::new(
        node.name.unwrap_or_default(),
        node.op_type.unwrap_or_default(),
        node.input,
        node.output,
        attributes,
    ))
}

/// Parse ONNX attribute into AttributeValue
///
/// Strings and string arrays are stored as `prost::bytes::Bytes` to avoid
/// mandatory UTF-8 validation during parsing. This allows zero-copy moves from
/// the protobuf structure.
pub(crate) fn parse_attribute_proto(
    attr: AttributeProto,
    external_data_loader: &Option<Arc<ExternalDataLoader>>,
) -> Result<AttributeValue, Error> {
    let attr_type = attr.r#type.unwrap_or(0);
    match attr_type {
        1 => Ok(AttributeValue::Float(attr.f.unwrap_or(0.0))),
        2 => Ok(AttributeValue::Int(attr.i.unwrap_or(0))),
        3 => Ok(AttributeValue::String(attr.s.unwrap_or_default())),
        4 => {
            if let Some(tensor) = attr.t {
                // Note: Tensor attributes don't have external data loader since they're inline
                let onnx_tensor = tensor_from_proto(tensor, &None)?;
                Ok(AttributeValue::Tensor(Box::new(onnx_tensor)))
            } else {
                Err(Error::MissingField("tensor attribute data".to_string()))
            }
        }
        5 => {
            if let Some(graph) = attr.g {
                let onnx_graph = Graph::from_proto(graph, external_data_loader.clone())?;
                Ok(AttributeValue::Graph(Box::new(onnx_graph)))
            } else {
                Err(Error::MissingField("graph attribute data".to_string()))
            }
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
                .map(|graph| Graph::from_proto(graph, external_data_loader.clone()))
                .collect::<Result<Box<[Graph]>, Error>>()?,
        )),
        _ => Err(Error::Unsupported(format!("attribute type: {}", attr_type))),
    }
}
