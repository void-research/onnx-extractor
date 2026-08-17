use crate::AttributeValue;
use std::collections::HashMap;

/// An ONNX operation/node in the computational graph
#[derive(Debug)]
pub struct Operation {
    name: String,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: HashMap<String, AttributeValue>,
}

impl Operation {
    pub(crate) fn new(
        name: String,
        op_type: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: HashMap<String, AttributeValue>,
    ) -> Self {
        Operation {
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

    /// Mutable reference to all attributes.
    ///
    /// To remove or drain attributes directly.
    pub fn attributes_mut(&mut self) -> &mut HashMap<String, AttributeValue> {
        &mut self.attributes
    }
}
