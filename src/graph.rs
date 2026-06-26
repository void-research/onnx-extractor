use std::collections::{HashMap, hash_map::Entry};
use std::sync::Arc;

use crate::external_data::ExternalDataLoader;
use crate::{AttributeValue, Error, GraphProto, Operation, Tensor, proto_adapter, type_proto};

/// Represents a computational graph in an ONNX model (either the root graph or a nested subgraph)
#[derive(Debug)]
pub struct Graph {
    name: String,
    tensors: HashMap<String, Tensor>,
    operations: Vec<Operation>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl Graph {
    pub(crate) fn from_proto(
        mut graph: GraphProto,
        external_data_loader: Option<Arc<ExternalDataLoader>>,
    ) -> Result<Self, Error> {
        let mut onnx_graph = Graph {
            name: graph.name.take().unwrap_or_default(),
            tensors: HashMap::new(),
            operations: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };

        // Pre-allocate based on graph sizes to avoid repeated reallocations
        onnx_graph.tensors.reserve(
            graph.initializer.len()
                + graph.value_info.len()
                + graph.input.len()
                + graph.output.len(),
        );
        onnx_graph.operations.reserve(graph.node.len());
        onnx_graph.inputs.reserve(graph.input.len());
        onnx_graph.outputs.reserve(graph.output.len());

        // Parse initialiser tensors (weights/constants) by draining to avoid clones
        for tensor in graph.initializer.drain(..) {
            let onnx_tensor = proto_adapter::tensor_from_proto(tensor, &external_data_loader)?;
            let tensor_name = onnx_tensor.name().to_string();
            onnx_graph.tensors.insert(tensor_name, onnx_tensor);
        }

        // Parse input tensor info and extract input names
        for mut input in graph.input.drain(..) {
            let name = input.name.take().unwrap_or_default();

            // If the name is already in tensors, it's an initializer, so we skip adding it to inputs/tensors
            if !onnx_graph.tensors.contains_key(&name) {
                if let Some(type_proto::Value::TensorType(tensor_type)) =
                    input.r#type.take().and_then(|t| t.value)
                {
                    onnx_graph.inputs.push(name.clone());
                    let onnx_tensor = Tensor::from_tensor_type(name.clone(), &tensor_type)?;
                    onnx_graph.tensors.insert(name, onnx_tensor);
                } else {
                    onnx_graph.inputs.push(name);
                }
            }
        }

        // Parse value_info for intermediate tensor shapes and types
        for mut value_info in graph.value_info.drain(..) {
            if let Some(type_proto::Value::TensorType(tensor_type)) =
                value_info.r#type.take().and_then(|t| t.value)
            {
                let name = value_info.name.take().unwrap_or_default();
                if let Entry::Vacant(e) = onnx_graph.tensors.entry(name) {
                    let onnx_tensor = Tensor::from_tensor_type(e.key().clone(), &tensor_type)?;
                    e.insert(onnx_tensor);
                }
            }
        }

        // Parse output tensor info and extract output names
        for mut output in graph.output.drain(..) {
            let name = output.name.take().unwrap_or_default();

            if !onnx_graph.tensors.contains_key(&name)
                && let Some(type_proto::Value::TensorType(tensor_type)) =
                    output.r#type.take().and_then(|t| t.value)
            {
                onnx_graph.outputs.push(name.clone());
                let onnx_tensor = Tensor::from_tensor_type(name.clone(), &tensor_type)?;
                onnx_graph.tensors.insert(name, onnx_tensor);
            } else {
                onnx_graph.outputs.push(name);
            }
        }

        // Parse operations/nodes by draining to allow owned conversion
        for node in graph.node.drain(..) {
            let operation = Operation::from_node_proto(node, &external_data_loader)?;
            onnx_graph.operations.push(operation);
        }

        Ok(onnx_graph)
    }

    /// Reference to all tensors in this graph
    pub fn tensors(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }

    /// Mutable reference to all tensors in this graph.
    ///
    /// To remove, drain, or modify tensors directly.
    pub fn tensors_mut(&mut self) -> &mut HashMap<String, Tensor> {
        &mut self.tensors
    }

    /// Get all operations in the graph
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Mutable reference to all operations in the graph.
    ///
    /// To remove, drain, or modify operations directly.
    pub fn operations_mut(&mut self) -> &mut Vec<Operation> {
        &mut self.operations
    }

    /// Get names of graph inputs
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    /// Get names of graph outputs
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }

    /// Get graph name
    pub fn graph_name(&self) -> &str {
        &self.name
    }

    /// Get all operations of a specific type in this graph
    pub fn get_operations_by_type(&self, op_type: &str) -> impl Iterator<Item = &Operation> {
        self.operations
            .iter()
            .filter(move |&op| op.op_type() == op_type)
    }

    /// Get operation by name in this graph
    pub fn get_operation(&self, name: &str) -> Option<&Operation> {
        self.operations.iter().find(|op| op.name() == name)
    }

    /// Get all operation types in this graph
    pub fn operation_types(&self) -> Box<[&str]> {
        let mut op_types: Vec<&str> = self.operations.iter().map(|op| op.op_type()).collect();
        op_types.sort_unstable();
        op_types.dedup();
        op_types.into_boxed_slice()
    }

    /// Count operations by type in this graph
    pub fn count_operations_by_type(&self) -> HashMap<&str, usize> {
        let mut counts = HashMap::new();
        for op in &self.operations {
            *counts.entry(op.op_type()).or_insert(0) += 1;
        }
        counts
    }

    /// Get input tensors in this graph
    pub fn get_input_tensors(&self) -> impl Iterator<Item = &Tensor> {
        self.inputs.iter().filter_map(|name| self.tensors.get(name))
    }

    /// Get output tensors in this graph
    pub fn get_output_tensors(&self) -> impl Iterator<Item = &Tensor> {
        self.outputs
            .iter()
            .filter_map(|name| self.tensors.get(name))
    }

    /// Get tensors with data (initialisers/weights) in this graph
    pub fn get_weight_tensors(&self) -> impl Iterator<Item = &Tensor> {
        self.tensors.values().filter(|&t| t.has_data())
    }

    /// Return operations in topological order using Kahn's algorithm.
    ///
    /// The returned vector contains references into `self.operations` and
    /// represents an order such that producers appear before their consumers.
    /// Operations are processed in the order they become available with no
    /// additional prioritisation.
    ///
    /// If the graph contains cycles or there are unresolved dependencies,
    /// the function returns an `Error::InvalidModel`.
    pub fn topological_order(&self) -> Result<Vec<&Operation>, Error> {
        let op_count = self.operations.len();

        // map tensor name -> producer op index
        let mut producer: HashMap<&str, usize> = HashMap::with_capacity(op_count);

        for (idx, op) in self.operations.iter().enumerate() {
            for out in op.outputs() {
                if !out.is_empty() {
                    producer.insert(out.as_str(), idx);
                }
            }
        }

        // list of consumer op indices for each producer op index
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); op_count];
        // indegree = number of inputs coming from other ops
        let mut indegree = vec![0; op_count];

        for (idx, op) in self.operations.iter().enumerate() {
            for input in op.inputs() {
                if !input.is_empty()
                    && let Some(&prod_idx) = producer.get(input.as_str())
                {
                    indegree[idx] += 1;
                    consumers[prod_idx].push(idx);
                }
            }
        }

        // start with ops that have indegree 0
        let mut stack: Vec<usize> = indegree
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 0)
            .map(|(idx, _)| idx)
            .collect();

        let mut ordered: Vec<&Operation> = Vec::with_capacity(op_count);

        while let Some(idx) = stack.pop() {
            let op = &self.operations[idx];
            ordered.push(op);

            for &cidx in &consumers[idx] {
                indegree[cidx] -= 1;
                if indegree[cidx] == 0 {
                    stack.push(cidx);
                }
            }
        }

        if ordered.len() != op_count {
            Err(Error::InvalidModel(
                "Graph has cycles or unresolved dependencies".to_string(),
            ))
        } else {
            Ok(ordered)
        }
    }

    /// Format graph information recursively
    pub(crate) fn format_recursive(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        indent_level: usize,
    ) -> std::fmt::Result {
        let indent = "  ".repeat(indent_level);
        let weight_count = self.get_weight_tensors().count();
        let op_counts = self.count_operations_by_type();
        writeln!(f, "{}ONNX Graph: {}", indent, self.name)?;
        writeln!(
            f,
            "{}Inputs: {} | Outputs: {} | Operations: {} | Tensors: {}",
            indent,
            self.inputs.len(),
            self.outputs.len(),
            self.operations.len(),
            self.tensors.len()
        )?;

        writeln!(f, "{}Operation types: {:?}", indent, op_counts)?;

        writeln!(f, "{}Weight tensors: {}", indent, weight_count)?;
        writeln!(f, "{}Input Names: {:?}", indent, self.inputs)?;
        writeln!(f, "{}Output Names: {:?}", indent, self.outputs)?;

        writeln!(f, "\n{}Tensors ({}):", indent, self.tensors.len())?;
        for (name, tensor) in &self.tensors {
            writeln!(
                f,
                "{}  {}: {:?} ({:?}) [{}{}]",
                indent,
                name,
                tensor.shape(),
                tensor.data_type(),
                if tensor.has_data() { "data" } else { "no data" },
                if self.inputs.contains(name) {
                    ", input"
                } else if self.outputs.contains(name) {
                    ", output"
                } else {
                    ""
                }
            )?;
        }

        writeln!(f, "\n{}Operations ({}):", indent, self.operations.len())?;
        for (op_type, count) in &op_counts {
            writeln!(f, "{}  {}: {} operations", indent, op_type, count)?;
        }

        writeln!(f, "\n{}Operation Details:", indent)?;
        for op in &self.operations {
            writeln!(
                f,
                "{}  {} ({}): {} -> {}",
                indent,
                op.name(),
                op.op_type(),
                op.inputs().join(", "),
                op.outputs().join(", ")
            )?;
            if !op.attributes().is_empty() {
                let attr_keys: Box<[_]> = op.attributes().keys().collect();
                writeln!(f, "{}    Attributes: {:?}", indent, attr_keys)?;

                for (attr_name, attr_val) in op.attributes() {
                    match attr_val {
                        AttributeValue::Graph(subgraph) => {
                            writeln!(f, "{}      {} (Subgraph):", indent, attr_name)?;
                            subgraph.format_recursive(f, indent_level + 4)?;
                        }
                        AttributeValue::Graphs(subgraphs) => {
                            writeln!(
                                f,
                                "{}      {} ({} subgraphs):",
                                indent,
                                attr_name,
                                subgraphs.len()
                            )?;
                            for (sub_idx, subgraph) in subgraphs.iter().enumerate() {
                                writeln!(f, "{}        [{}]:", indent, sub_idx)?;
                                subgraph.format_recursive(f, indent_level + 6)?;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format_recursive(f, 0)
    }
}
