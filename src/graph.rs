use std::collections::{HashMap, HashSet};

use crate::{AttributeValue, Error, Operation, Tensor};

/// Represents a computational graph in an ONNX model
#[derive(Debug)]
pub struct Graph {
    name: Option<String>,
    tensors: HashMap<String, Tensor>,
    operations: Vec<Operation>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl Graph {
    pub(crate) fn new(
        name: Option<String>,
        tensors: HashMap<String, Tensor>,
        operations: Vec<Operation>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Graph {
            name,
            tensors,
            operations,
            inputs,
            outputs,
        }
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
    pub fn graph_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get all operations of a specific type in this graph
    pub fn get_operations_by_type(&self, op_type: &str) -> impl Iterator<Item = &Operation> {
        self.operations
            .iter()
            .filter(move |&op| op.op_type() == op_type)
    }

    /// Get operation by name in this graph
    pub fn get_operation(&self, name: &str) -> Option<&Operation> {
        self.operations.iter().find(|op| op.name() == Some(name))
    }

    /// Get all operation types in this graph
    pub fn operation_types(&self) -> HashSet<&str> {
        self.operations.iter().map(Operation::op_type).collect()
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
    /// the function returns an `Error::InvalidGraph`.
    pub fn topological_order(&self) -> Result<Vec<&Operation>, Error> {
        let op_count = self.operations.len();

        // map tensor name -> producer op index
        let producer: HashMap<&str, usize> = self
            .operations
            .iter()
            .enumerate()
            .flat_map(|(idx, op)| {
                op.outputs()
                    .iter()
                    .filter(|s| !s.is_empty())
                    .map(move |s| (s.as_str(), idx))
            })
            .collect();

        // build consumer lists and in-degree counts
        let mut consumers = vec![Vec::new(); op_count];
        let mut indegree = vec![0; op_count];

        for (idx, op) in self.operations.iter().enumerate() {
            for input in op.inputs().iter().filter(|s| !s.is_empty()) {
                if let Some(&prod_idx) = producer.get(input.as_str()) {
                    indegree[idx] += 1;
                    consumers[prod_idx].push(idx);
                }
            }
        }

        // start with ops that have indegree 0
        let mut stack: Vec<usize> = indegree
            .iter()
            .enumerate()
            .filter_map(|(idx, &d)| (d == 0).then_some(idx))
            .collect();

        let mut ordered = Vec::with_capacity(op_count);

        while let Some(idx) = stack.pop() {
            ordered.push(&self.operations[idx]);
            for &cidx in &consumers[idx] {
                indegree[cidx] -= 1;
                if indegree[cidx] == 0 {
                    stack.push(cidx);
                }
            }
        }

        if ordered.len() == op_count {
            Ok(ordered)
        } else {
            Err(Error::InvalidGraph)
        }
    }

    pub(crate) fn format_recursive(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        indent_level: usize,
    ) -> std::fmt::Result {
        let indent = "  ".repeat(indent_level);
        let weight_count = self.get_weight_tensors().count();
        let op_counts = self.count_operations_by_type();

        writeln!(f, "{indent}ONNX Graph: {:?}", self.name)?;
        writeln!(
            f,
            "{indent}Inputs: {} | Outputs: {} | Operations: {} | Tensors: {}",
            self.inputs.len(),
            self.outputs.len(),
            self.operations.len(),
            self.tensors.len()
        )?;

        writeln!(f, "{indent}Operation types: {op_counts:?}")?;
        writeln!(f, "{indent}Weight tensors: {weight_count}")?;
        writeln!(f, "{indent}Input Names: {:?}", self.inputs)?;
        writeln!(f, "{indent}Output Names: {:?}", self.outputs)?;

        writeln!(f, "\n{indent}Tensors ({}):", self.tensors.len())?;
        for (name, tensor) in &self.tensors {
            let role = if self.inputs.contains(name) {
                ", input"
            } else if self.outputs.contains(name) {
                ", output"
            } else {
                ""
            };
            let status = if tensor.has_data() { "data" } else { "no data" };
            writeln!(
                f,
                "{indent}  {name}: {:?} ({:?}) [{status}{role}]",
                tensor.shape(),
                tensor.data_type()
            )?;
        }

        writeln!(f, "\n{indent}Operations ({}):", self.operations.len())?;
        for (op_type, count) in &op_counts {
            writeln!(f, "{indent}  {op_type}: {count} operations")?;
        }

        writeln!(f, "\n{indent}Operation Details:")?;
        for op in &self.operations {
            writeln!(
                f,
                "{indent}  {:?} ({}): {} -> {}",
                op.name(),
                op.op_type(),
                op.inputs().join(", "),
                op.outputs().join(", ")
            )?;

            if !op.attributes().is_empty() {
                let attr_keys: Box<[&str]> = op.attributes().keys().map(String::as_str).collect();
                writeln!(f, "{indent}    Attributes: {attr_keys:?}")?;

                for (attr_name, attr_val) in op.attributes() {
                    match attr_val {
                        AttributeValue::Graph(subgraph) => {
                            writeln!(f, "{indent}      {attr_name} (Subgraph):")?;
                            subgraph.format_recursive(f, indent_level + 4)?;
                        }
                        AttributeValue::Graphs(subgraphs) => {
                            writeln!(
                                f,
                                "{indent}      {attr_name} ({} subgraphs):",
                                subgraphs.len()
                            )?;
                            for (sub_idx, subgraph) in subgraphs.iter().enumerate() {
                                writeln!(f, "{indent}        [{sub_idx}]:")?;
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
