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
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let bytes = Bytes::from_owner(mmap);

        // Extract model directory for external data loading
        let model_dir = path
            .parent()
            .map(Path::to_path_buf)
            .or_else(|| std::env::current_dir().ok());

        Self::load_from_bytes_with_dir_bytes(bytes, model_dir)
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

        let metadata: HashMap<String, String> = model
            .metadata_props
            .into_iter()
            .filter_map(|prop| prop.key.map(|k| (k, prop.value.unwrap_or_default())))
            .collect();

        let opsets: HashMap<String, i64> = model
            .opset_import
            .into_iter()
            .map(|opset| (opset.domain.unwrap_or_default(), opset.version.unwrap_or(0)))
            .collect();

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
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ONNX Model Information:")?;
        writeln!(f, "IR Version: {}", self.ir_version)?;
        writeln!(f, "Producer Name: {}", self.producer_name)?;
        writeln!(f, "Producer Version: {}", self.producer_version)?;
        writeln!(f, "Domain: {}", self.domain)?;
        writeln!(f, "Model Version: {}", self.model_version)?;
        writeln!(f, "Description: {}", self.doc_string)?;
        writeln!(f, "Metadata: {:?}", self.metadata)?;
        writeln!(f, "Opsets: {:?}", self.opsets)?;

        writeln!(f, "\nRoot Graph & Subgraph Details:")?;
        write!(f, "{}", self.graph)
    }
}
