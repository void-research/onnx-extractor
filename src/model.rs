use memmap2::Mmap;
use prost::Message;
use prost::bytes::Bytes;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::external_data::ExternalDataLoader;
use crate::{Error, ModelProto, OnnxOperation, OnnxTensor, proto_adapter, type_proto};

/// Main ONNX model container
pub struct OnnxModel {
    pub tensors: HashMap<String, OnnxTensor>,
    pub operations: Vec<OnnxOperation>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub model_version: i64,
    pub producer_name: String,
    pub producer_version: String,
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
        // Tensors keep the loader alive via Rc as long as they need it
        let external_data_loader = model_dir.map(|dir| Rc::new(ExternalDataLoader::new(dir)));

        let mut onnx_model = OnnxModel {
            tensors: HashMap::new(),
            operations: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            model_version: model.model_version.unwrap_or(0),
            producer_name: model.producer_name.unwrap_or_default(),
            producer_version: model.producer_version.unwrap_or_default(),
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
        for input in graph.input.drain(..) {
            let name = input.name.clone().unwrap_or_default();
            if name.is_empty() {
                continue;
            }

            // If the name is already in tensors, it's an initialiser, so we skip adding it to inputs
            if !onnx_model.tensors.contains_key(&name) {
                onnx_model.inputs.push(name.clone());
            }

            if let Some(t) = &input.r#type
                && let Some(type_proto_value) = &t.value
                && let type_proto::Value::TensorType(tensor_type) = type_proto_value
            {
                let onnx_tensor = OnnxTensor::from_tensor_type(name.clone(), tensor_type)?;
                onnx_model.tensors.entry(name).or_insert(onnx_tensor);
            }
        }

        // parse value_info for intermediate tensor shapes and types
        for value_info in graph.value_info.drain(..) {
            if let Some(t) = &value_info.r#type
                && let Some(type_proto_value) = &t.value
                && let type_proto::Value::TensorType(tensor_type) = type_proto_value
            {
                let name = value_info.name.unwrap_or_default();
                if !name.is_empty() {
                    let onnx_tensor = OnnxTensor::from_tensor_type(name.clone(), tensor_type)?;
                    onnx_model.tensors.entry(name).or_insert(onnx_tensor);
                }
            }
        }

        // parse output tensor info and extract output names
        for output in graph.output.drain(..) {
            let name = output.name.clone().unwrap_or_default();
            if name.is_empty() {
                continue;
            }

            onnx_model.outputs.push(name.clone());

            if let Some(t) = &output.r#type
                && let Some(type_proto_value) = &t.value
                && let type_proto::Value::TensorType(tensor_type) = type_proto_value
            {
                let onnx_tensor = OnnxTensor::from_tensor_type(name.clone(), tensor_type)?;
                onnx_model.tensors.entry(name).or_insert(onnx_tensor);
            }
        }

        // parse operations/nodes by draining to allow owned conversion
        for node in graph.node.drain(..) {
            let operation = OnnxOperation::from_node_proto(node)?;
            onnx_model.operations.push(operation);
        }

        Ok(onnx_model)
    }

    /// Get tensor information by name
    pub fn get_tensor(&self, name: &str) -> Option<&OnnxTensor> {
        self.tensors.get(name)
    }

    /// Get all operations of a specific type
    pub fn get_operations_by_type(&self, op_type: &str) -> Vec<&OnnxOperation> {
        self.operations
            .iter()
            .filter(|op| op.op_type == op_type)
            .collect()
    }

    /// Get operation by name
    pub fn get_operation(&self, name: &str) -> Option<&OnnxOperation> {
        self.operations.iter().find(|op| op.name == name)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Get all operation types in the model
    pub fn operation_types(&self) -> Vec<String> {
        // collect unique operation types using a hash set of &str to avoid
        // allocating intermediate owned Strings, then sort the resulting Vec
        let mut set: HashSet<&str> = HashSet::with_capacity(self.operations.len());
        for op in &self.operations {
            set.insert(op.op_type.as_str());
        }
        let mut op_types: Vec<String> = set.into_iter().map(|s| s.to_string()).collect();
        op_types.sort();
        op_types
    }

    /// Count operations by type
    pub fn count_operations_by_type(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        counts.reserve(self.operations.len());
        for op in &self.operations {
            *counts.entry(op.op_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Get input tensors
    pub fn get_input_tensors(&self) -> Vec<&OnnxTensor> {
        self.inputs
            .iter()
            .filter_map(|name| self.get_tensor(name))
            .collect()
    }

    /// Get output tensors
    pub fn get_output_tensors(&self) -> Vec<&OnnxTensor> {
        self.outputs
            .iter()
            .filter_map(|name| self.get_tensor(name))
            .collect()
    }

    /// Get tensors with data (initialisers/weights)
    pub fn get_weight_tensors(&self) -> Vec<&OnnxTensor> {
        self.tensors
            .values()
            .filter(|tensor| tensor.data().is_ok())
            .collect()
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
        let mut producer: HashMap<&str, usize> = HashMap::new();
        for (idx, op) in self.operations.iter().enumerate() {
            for out in &op.outputs {
                if !out.is_empty() {
                    producer.entry(out.as_str()).or_insert(idx);
                }
            }
        }

        // map tensor name -> list of consumer op indices
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, op) in self.operations.iter().enumerate() {
            for input in &op.inputs {
                if !input.is_empty() {
                    consumers.entry(input.as_str()).or_default().push(idx);
                }
            }
        }

        // indegree = number of inputs coming from other ops (i.e. produced by some op)
        let mut indegree: Vec<usize> = vec![0; op_count];
        for (idx, op) in self.operations.iter().enumerate() {
            let mut count = 0usize;
            for input in &op.inputs {
                if input.is_empty() {
                    continue;
                }
                if producer.contains_key(input.as_str()) {
                    count += 1;
                }
            }
            indegree[idx] = count;
        }

        // start with ops that have indegree 0
        let mut queue: VecDeque<usize> = VecDeque::new();
        for (idx, &d) in indegree.iter().enumerate() {
            if d == 0 {
                queue.push_back(idx);
            }
        }

        let mut ordered: Vec<&OnnxOperation> = Vec::with_capacity(op_count);

        while let Some(idx) = queue.pop_front() {
            let op = &self.operations[idx];
            ordered.push(op);

            // mark outputs as available and reduce indegree of consumers
            for out in &op.outputs {
                if out.is_empty() {
                    continue;
                }
                if let Some(cons_list) = consumers.get(out.as_str()) {
                    for &cidx in cons_list {
                        // only decrease indegree if the dependency was counted from a producer
                        if indegree[cidx] > 0 {
                            indegree[cidx] -= 1;
                            if indegree[cidx] == 0 {
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
        let mut producer: HashMap<&str, usize> = HashMap::new();
        for (idx, op) in self.operations.iter().enumerate() {
            for out in &op.outputs {
                if !out.is_empty() {
                    producer.entry(out.as_str()).or_insert(idx);
                }
            }
        }

        // map tensor name -> list of consumer op indices
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, op) in self.operations.iter().enumerate() {
            for input in &op.inputs {
                if !input.is_empty() {
                    consumers.entry(input.as_str()).or_default().push(idx);
                }
            }
        }

        // indegree = number of inputs coming from other ops
        let mut indegree: Vec<usize> = vec![0; op_count];
        for (idx, op) in self.operations.iter().enumerate() {
            let mut count = 0usize;
            for input in &op.inputs {
                if input.is_empty() {
                    continue;
                }
                if producer.contains_key(input.as_str()) {
                    count += 1;
                }
            }
            indegree[idx] = count;
        }

        // start with ops that have indegree 0, prioritizing those that consume model inputs
        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut ready_ops: Vec<usize> = Vec::new();

        for (idx, &d) in indegree.iter().enumerate() {
            if d == 0 {
                ready_ops.push(idx);
            }
        }

        // sort ready ops: input consumers first, then parameter-only ops
        ready_ops.sort_by_key(|&idx| {
            let op = &self.operations[idx];
            let consumes_input = op.inputs.iter().any(|input| self.inputs.contains(input));
            !consumes_input // false sorts before true, so input consumers come first
        });

        for idx in ready_ops {
            queue.push_back(idx);
        }

        let mut ordered: Vec<&OnnxOperation> = Vec::with_capacity(op_count);

        while let Some(idx) = queue.pop_front() {
            let op = &self.operations[idx];
            ordered.push(op);

            // collect newly ready operations
            let mut newly_ready: Vec<usize> = Vec::new();

            for out in &op.outputs {
                if out.is_empty() {
                    continue;
                }
                if let Some(cons_list) = consumers.get(out.as_str()) {
                    for &cidx in cons_list {
                        if indegree[cidx] > 0 {
                            indegree[cidx] -= 1;
                            if indegree[cidx] == 0 {
                                newly_ready.push(cidx);
                            }
                        }
                    }
                }
            }

            // sort newly ready ops: input consumers first
            newly_ready.sort_by_key(|&idx| {
                let op = &self.operations[idx];
                let consumes_input = op.inputs.iter().any(|input| self.inputs.contains(input));
                !consumes_input
            });

            // add to front of queue (input consumers) or back (parameter ops)
            for cidx in newly_ready {
                let consumer_op = &self.operations[cidx];
                let consumes_input = consumer_op
                    .inputs
                    .iter()
                    .any(|input| self.inputs.contains(input));

                if consumes_input {
                    queue.push_front(cidx);
                } else {
                    queue.push_back(cidx);
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
            "Producer: {} v{}",
            self.producer_name, self.producer_version
        );
        println!("Model Version: {}", self.model_version);
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
                op.name,
                op.op_type,
                op.inputs.join(", "),
                op.outputs.join(", ")
            );
            if !op.attributes.is_empty() {
                println!("    Attributes: {:?}", op.attribute_names());
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

        let weight_count = self.get_weight_tensors().len();
        println!("Weight tensors: {}", weight_count);
    }
}
