use crate::{AttributeValue, Error, NodeProto, proto_adapter};
use std::collections::HashMap;

/// Information about an ONNX operation/node
#[derive(Debug)]
pub struct OnnxOperation {
    name: String,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: HashMap<String, AttributeValue>,
}

impl OnnxOperation {
    pub(crate) fn new(
        name: String,
        op_type: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: HashMap<String, AttributeValue>,
    ) -> Self {
        OnnxOperation {
            name,
            op_type,
            inputs,
            outputs,
            attributes,
        }
    }
    /// Operation name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Operation type (e.g., "Conv", "Relu")
    pub fn op_type(&self) -> &str {
        &self.op_type
    }

    /// Input tensor names
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    /// Output tensor names
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }

    /// Reference to all attributes
    pub fn attributes(&self) -> &HashMap<String, AttributeValue> {
        &self.attributes
    }

    /// Create OnnxOperation from ONNX NodeProto
    pub(crate) fn from_node_proto(node: NodeProto) -> Result<Self, Error> {
        proto_adapter::operation_from_node_proto(node)
    }
}
