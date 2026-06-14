use memmap2::Mmap;
use prost::Message;
use prost::bytes::Bytes;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::external_data::ExternalDataLoader;
use crate::{Error, Graph, ModelProto};

/// Main ONNX model container
pub struct Model {
    graph: Graph,
    ir_version: i64,
    producer_name: String,
    producer_version: String,
    domain: String,
    model_version: i64,
    doc_string: String,
    metadata: HashMap<String, String>,
    opsets: HashMap<String, i64>,
}

impl Model {
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
        let graph = model
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

        Ok(Model {
            graph: Graph::from_proto(graph, external_data_loader)?,
            ir_version: model.ir_version.unwrap_or(0),
            producer_name: model.producer_name.unwrap_or_default(),
            producer_version: model.producer_version.unwrap_or_default(),
            domain: model.domain.unwrap_or_default(),
            model_version: model.model_version.unwrap_or(0),
            doc_string: model.doc_string.unwrap_or_default(),
            metadata,
            opsets,
        })
    }

    /// Reference to the root graph of the model
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Mutable reference to the root graph of the model
    pub fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    /// Consume the model and return the root graph
    pub fn into_graph(self) -> Graph {
        self.graph
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

    /// Get custom metadata properties
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Get operator set imports (domain -> version)
    pub fn opsets(&self) -> &HashMap<String, i64> {
        &self.opsets
    }

    /// Print comprehensive model information and recursive graph tree
    pub fn print_model_info(&self) {
        println!("=== ONNX Model Information ===");
        println!(
            "Producer: {} v{} (IR v{}, Domain: {})",
            self.producer_name, self.producer_version, self.ir_version, self.domain
        );
        println!("Model Version: {}", self.model_version,);

        if !self.doc_string.is_empty() {
            println!("Description: {}", self.doc_string);
        }
        if !self.metadata.is_empty() {
            println!("Metadata: {:?}", self.metadata);
        }
        if !self.opsets.is_empty() {
            println!("Opset Imports: {:?}", self.opsets);
        }

        println!("\nRoot Graph & Subgraph Details:");
        self.graph.print_graph_info_recursive(0, true);
    }
}
