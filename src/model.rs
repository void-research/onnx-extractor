use memmap2::Mmap;
use prost::Message;
use prost::bytes::Bytes;
use std::collections::hash_map::Drain;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::external_data::ExternalDataLoader;
use crate::{Error, ModelProto, OnnxOperation, OnnxTensor, proto_adapter, type_proto};

/// Main ONNX model container
pub struct OnnxModel {
    tensors: HashMap<String, OnnxTensor>,
    operations: Vec<OnnxOperation>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    ir_version: i64,
    producer_name: String,
    producer_version: String,
    domain: String,
    model_version: i64,
    doc_string: String,
    graph_name: String,
    metadata: HashMap<String, String>,
    opsets: HashMap<String, i64>,
}

impl OnnxModel {
    /// Load ONNX model from file path
    pub fn load_from_file(path: &str) -> Result<Self, Error> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let bytes = Bytes::from_owner(mmap);

        // Extract model directory for external data loading
        let model_dir = Path::new(path)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        Self::load_from_bytes_with_dir_bytes(bytes, Some(model_dir))
    }

    /// Load ONNX model from owned byte vector
    pub fn load_from_bytes(data: Vec<u8>) -> Result<Self, Error> {
        Self::load_from_bytes_with_dir_bytes(Bytes::from(data), None)
    }

    /// Load ONNX model from owned byte vector with optional model directory for external data
    fn load_from_bytes_with_dir_bytes(
        data: Bytes,
        model_dir: Option<PathBuf>,
    ) -> Result<Self, Error> {
        let model = ModelProto::decode(data)?;
        let mut graph = model
            .graph
            .ok_or_else(|| Error::InvalidModel("No graph found in model".to_string()))?;

        // Create external data loader if model directory is available
        // Tensors keep the loader alive via Arc as long as they need it
        let external_data_loader = model_dir.map(|dir| Arc::new(ExternalDataLoader::new(dir)));

        let mut metadata = HashMap::new();
        for mut prop in model.metadata_props {
            if let Some(key) = prop.key.take() {
                metadata.insert(key, prop.value.take().unwrap_or_default());
            }
        }

        let mut opsets = HashMap::new();
        for mut opset in model.opset_import {
            opsets.insert(
                opset.domain.take().unwrap_or_default(),
                opset.version.unwrap_or(0),
            );
        }

        let mut onnx_model = OnnxModel {
            tensors: HashMap::new(),
            operations: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            ir_version: model.ir_version.unwrap_or(0),
            producer_name: model.producer_name.unwrap_or_default(),
            producer_version: model.producer_version.unwrap_or_default(),
            domain: model.domain.unwrap_or_default(),
            model_version: model.model_version.unwrap_or(0),
            doc_string: model.doc_string.unwrap_or_default(),
            graph_name: graph.name.clone().unwrap_or_default(),
            metadata,
            opsets,
        };

        // pre-allocate based on graph sizes to avoid repeated reallocations
        onnx_model.tensors.reserve(
            graph.initializer.len()
                + graph.value_info.len()
                + graph.input.len()
                + graph.output.len(),
        );
        onnx_model.operations.reserve(graph.node.len());
        onnx_model.inputs.reserve(graph.input.len());
        onnx_model.outputs.reserve(graph.output.len());

        // parse initialiser tensors (weights/constants) by draining to avoid clones
        for tensor in graph.initializer.drain(..) {
            let onnx_tensor =
                proto_adapter::tensor_from_proto(tensor, external_data_loader.clone())?;
            let tensor_name = onnx_tensor.name().to_string();
            if !tensor_name.is_empty() {
                onnx_model.tensors.insert(tensor_name, onnx_tensor);
            }
        }

        // parse input tensor info and extract input names
        for mut input in graph.input.drain(..) {
            let name = input.name.take().unwrap_or_default();
            if name.is_empty() {
                continue;
            }

            // If the name is already in tensors, it's an initialiser, so we skip adding it to inputs
            if !onnx_model.tensors.contains_key(&name) {
                onnx_model.inputs.push(name.clone());
            }

            if let Some(type_proto::Value::TensorType(tensor_type)) =
                input.r#type.take().and_then(|t| t.value)
                && !onnx_model.tensors.contains_key(&name)
            {
                let onnx_tensor = OnnxTensor::from_tensor_type(name.clone(), &tensor_type)?;
                onnx_model.tensors.insert(name, onnx_tensor);
            }
        }

        // parse value_info for intermediate tensor shapes and types
        for mut value_info in graph.value_info.drain(..) {
            if let Some(type_proto::Value::TensorType(tensor_type)) =
                value_info.r#type.take().and_then(|t| t.value)
            {
                let name = value_info.name.take().unwrap_or_default();
                if !name.is_empty() && !onnx_model.tensors.contains_key(&name) {
                    let onnx_tensor = OnnxTensor::from_tensor_type(name.clone(), &tensor_type)?;
                    onnx_model.tensors.insert(name, onnx_tensor);
                }
            }
        }

        // parse output tensor info and extract output names
        for mut output in graph.output.drain(..) {
            let name = output.name.take().unwrap_or_default();
            if name.is_empty() {
                continue;
            }

            onnx_model.outputs.push(name.clone());

            if let Some(type_proto::Value::TensorType(tensor_type)) =
                output.r#type.take().and_then(|t| t.value)
                && !onnx_model.tensors.contains_key(&name)
            {
                let onnx_tensor = OnnxTensor::from_tensor_type(name.clone(), &tensor_type)?;
                onnx_model.tensors.insert(name, onnx_tensor);
            }
        }

        // parse operations/nodes by draining to allow owned conversion
        for node in graph.node.drain(..) {
            let operation = OnnxOperation::from_node_proto(node)?;
            onnx_model.operations.push(operation);
        }

        Ok(onnx_model)
    }

    /// Reference to all tensors
    pub fn tensors(&self) -> &HashMap<String, OnnxTensor> {
        &self.tensors
    }

    /// Consume the model and return the underlying tensor map.
    pub fn into_tensors(self) -> HashMap<String, OnnxTensor> {
        self.tensors
    }

    /// Pluck a single tensor out of the model by name, taking ownership.
    /// This allows zero-copy extraction via OnnxTensor::into_data().
    pub fn take_tensor(&mut self, name: &str) -> Option<OnnxTensor> {
        self.tensors.remove(name)
    }

    /// Drain all tensors from the model, returning an iterator that takes ownership.
    /// The model remains alive but its tensor storage is cleared.
    pub fn drain_tensors(&mut self) -> Drain<'_, String, OnnxTensor> {
        self.tensors.drain()
    }

    /// Get all operations in the model
    pub fn operations(&self) -> &[OnnxOperation] {
        &self.operations
    }

    /// Get names of model inputs
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    /// Get names of model outputs
    pub fn outputs(&self) -> &[String] {
        &self.outputs
    }

    /// Get IR version
    pub fn ir_version(&self) -> i64 {
        self.ir_version
    }

    /// Get producer name
    pub fn producer_name(&self) -> &str {
        &self.producer_name
    }

    /// Get producer version
    pub fn producer_version(&self) -> &str {
        &self.producer_version
    }

    /// Get model domain
    pub fn domain(&self) -> &str {
        &self.domain
    }

    /// Get model version
    pub fn model_version(&self) -> i64 {
        self.model_version
    }

    /// Get documentation string
    pub fn doc_string(&self) -> &str {
        &self.doc_string
    }

    /// Get graph name
    pub fn graph_name(&self) -> &str {
        &self.graph_name
    }

    /// Get custom metadata properties
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Get operator set imports (domain -> version)
    pub fn opsets(&self) -> &HashMap<String, i64> {
        &self.opsets
    }

    /// Get all operations of a specific type
    pub fn get_operations_by_type(&self, op_type: &str) -> impl Iterator<Item = &OnnxOperation> {
        self.operations
            .iter()
            .filter(move |&op| op.op_type() == op_type)
    }

    /// Get operation by name
    pub fn get_operation(&self, name: &str) -> Option<&OnnxOperation> {
        self.operations.iter().find(|op| op.name() == name)
    }

    /// Get all operation types in the model
    pub fn operation_types(&self) -> Vec<String> {
        let mut set: HashSet<&str> = HashSet::new();
        for op in &self.operations {
            set.insert(op.op_type());
        }
        let mut op_types: Vec<String> = set.into_iter().map(|s| s.to_string()).collect();
        op_types.sort_unstable();
        op_types
    }

    /// Count operations by type
    pub fn count_operations_by_type(&self) -> HashMap<&str, usize> {
        let mut counts = HashMap::new();
        for op in &self.operations {
            *counts.entry(op.op_type()).or_insert(0) += 1;
        }
        counts
    }

    /// Get input tensors
    pub fn get_input_tensors(&self) -> impl Iterator<Item = &OnnxTensor> {
        self.inputs.iter().filter_map(|name| self.tensors.get(name))
    }

    /// Get output tensors
    pub fn get_output_tensors(&self) -> impl Iterator<Item = &OnnxTensor> {
        self.outputs
            .iter()
            .filter_map(|name| self.tensors.get(name))
    }

    /// Get tensors with data (initialisers/weights)
    pub fn get_weight_tensors(&self) -> impl Iterator<Item = &OnnxTensor> {
        self.tensors.values().filter(|&t| t.data().is_ok())
    }

    /// Return operations in a simple topological order using Kahn's algorithm.
    ///
    /// The returned vector contains references into `self.operations` and
    /// represents an order such that producers appear before their consumers.
    /// Operations are processed in the order they become available with no
    /// additional prioritisation.
    ///
    /// See also [`execution_order`](Self::execution_order) for a version that
    /// prioritises operations consuming model inputs.
    ///
    /// If the graph contains cycles or there are unresolved dependencies,
    /// the function returns an `Error::InvalidModel`.
    pub fn topological_order(&self) -> Result<Vec<&OnnxOperation>, Error> {
        let op_count = self.operations.len();

        // map tensor name -> producer op index
        let mut producer: HashMap<&str, usize> = HashMap::with_capacity(op_count);
        // map tensor name -> list of consumer op indices
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::with_capacity(op_count);

        for (idx, op) in self.operations.iter().enumerate() {
            for out in op.outputs() {
                if !out.is_empty() {
                    producer.insert(out.as_str(), idx);
                }
            }
            for input in op.inputs() {
                if !input.is_empty() {
                    consumers.entry(input.as_str()).or_default().push(idx);
                }
            }
        }

        // indegree = number of inputs coming from other ops
        let mut indegree = vec![0; op_count];
        for (idx, op) in self.operations.iter().enumerate() {
            let mut count = 0;
            for input in op.inputs() {
                if !input.is_empty() && producer.contains_key(input.as_str()) {
                    count += 1;
                }
            }
            indegree[idx] = count;
        }

        // start with ops that have indegree 0
        let mut queue: VecDeque<usize> = indegree
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 0)
            .map(|(idx, _)| idx)
            .collect();

        let mut ordered: Vec<&OnnxOperation> = Vec::with_capacity(op_count);

        while let Some(idx) = queue.pop_front() {
            let op = &self.operations[idx];
            ordered.push(op);

            for out in op.outputs() {
                if !out.is_empty()
                    && let Some(cons_list) = consumers.get(out.as_str())
                {
                    for &cidx in cons_list {
                        indegree[cidx] -= 1;
                        if indegree[cidx] == 0 {
                            queue.push_back(cidx);
                        }
                    }
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

    /// Return operations in execution-optimised topological order.
    ///
    /// The returned vector contains references into `self.operations` and
    /// represents an order such that producers appear before their consumers.
    /// Operations that consume model inputs are prioritised over parameter-only
    /// operations.
    ///
    /// This uses Kahn's algorithm with prioritisation. See also
    /// [`topological_order`](Self::topological_order) for a simple version
    /// without prioritisation.
    ///
    /// If the graph contains cycles or there are unresolved dependencies,
    /// the function returns an `Error::InvalidModel`.
    pub fn execution_order(&self) -> Result<Vec<&OnnxOperation>, Error> {
        let op_count = self.operations.len();

        // map tensor name -> producer op index
        let mut producer: HashMap<&str, usize> = HashMap::with_capacity(op_count);
        // map tensor name -> list of consumer op indices
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::with_capacity(op_count);

        for (idx, op) in self.operations.iter().enumerate() {
            for out in op.outputs() {
                if !out.is_empty() {
                    producer.insert(out.as_str(), idx);
                }
            }
            for input in op.inputs() {
                if !input.is_empty() {
                    consumers.entry(input.as_str()).or_default().push(idx);
                }
            }
        }

        // indegree = number of inputs coming from other ops
        let mut indegree = vec![0; op_count];
        for (idx, op) in self.operations.iter().enumerate() {
            let mut count = 0;
            for input in op.inputs() {
                if !input.is_empty() && producer.contains_key(input.as_str()) {
                    count += 1;
                }
            }
            indegree[idx] = count;
        }

        // start with ops that have indegree 0, prioritizing those that consume model inputs
        let mut ready_ops: Vec<usize> = indegree
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 0)
            .map(|(idx, _)| idx)
            .collect();

        // sort ready ops: input consumers first
        ready_ops.sort_by_key(|&idx| {
            let op = &self.operations[idx];
            let consumes_input = op.inputs().iter().any(|input| self.inputs.contains(input));
            !consumes_input
        });

        let mut queue: VecDeque<usize> = ready_ops.into();
        let mut ordered: Vec<&OnnxOperation> = Vec::with_capacity(op_count);

        while let Some(idx) = queue.pop_front() {
            let op = &self.operations[idx];
            ordered.push(op);

            for out in op.outputs() {
                if !out.is_empty()
                    && let Some(cons_list) = consumers.get(out.as_str())
                {
                    for &cidx in cons_list {
                        indegree[cidx] -= 1;
                        if indegree[cidx] == 0 {
                            let consumer_op = &self.operations[cidx];
                            let consumes_input = consumer_op
                                .inputs()
                                .iter()
                                .any(|input| self.inputs.contains(input));

                            if consumes_input {
                                queue.push_front(cidx);
                            } else {
                                queue.push_back(cidx);
                            }
                        }
                    }
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

    /// Print comprehensive model information
    pub fn print_model_info(&self) {
        println!("=== ONNX Model Information ===");
        println!(
            "Producer: {} v{} (IR v{}, Domain: {})",
            self.producer_name, self.producer_version, self.ir_version, self.domain
        );
        println!(
            "Model Version: {}, Graph Name: {}",
            self.model_version, self.graph_name
        );
        if !self.doc_string.is_empty() {
            println!("Description: {}", self.doc_string);
        }
        if !self.metadata.is_empty() {
            println!("Metadata: {:?}", self.metadata);
        }
        if !self.opsets.is_empty() {
            println!("Opset Imports: {:?}", self.opsets);
        }
        println!("Inputs: {:?}", self.inputs);
        println!("Outputs: {:?}", self.outputs);

        println!("\n=== Tensors ({}) ===", self.tensors.len());
        for (name, tensor) in &self.tensors {
            println!(
                "  {}: {:?} ({:?}) [{}{}]",
                name,
                tensor.shape(),
                tensor.data_type(),
                if tensor.data().is_ok() {
                    "data"
                } else {
                    "no data"
                },
                if self.inputs.contains(name) {
                    ", input"
                } else if self.outputs.contains(name) {
                    ", output"
                } else {
                    ""
                }
            );
        }

        println!("\n=== Operations ({}) ===", self.operations.len());
        let op_counts = self.count_operations_by_type();
        for (op_type, count) in &op_counts {
            println!("  {}: {} operations", op_type, count);
        }

        println!("\n=== Operation Details ===");
        for op in &self.operations {
            println!(
                "  {} ({}): {} -> {}",
                op.name(),
                op.op_type(),
                op.inputs().join(", "),
                op.outputs().join(", ")
            );
            if !op.attributes().is_empty() {
                println!("    Attributes: {:?}", op.attributes().keys());
            }
        }
    }

    /// Print a summary of the model
    pub fn print_summary(&self) {
        println!("=== ONNX Model Summary ===");
        println!(
            "Inputs: {} | Outputs: {} | Operations: {} | Tensors: {}",
            self.inputs.len(),
            self.outputs.len(),
            self.operations.len(),
            self.tensors.len()
        );

        let op_counts = self.count_operations_by_type();
        println!("Operation types: {:?}", op_counts);

        let weight_count = self.get_weight_tensors().count();
        println!("Weight tensors: {}", weight_count);
    }
}
