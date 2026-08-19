use memmap2::Mmap;
use prost::Message;
use prost::bytes::Bytes;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::external_data::ExternalDataLoader;
use crate::{Error, Graph, ModelProto, proto_adapter};

/// Main ONNX model container
pub struct Model {
    graph: Graph,
    ir_version: i64,
    producer_name: Option<String>,
    producer_version: Option<String>,
    domain: Option<String>,
    model_version: Option<i64>,
    doc_string: Option<String>,
    metadata: HashMap<String, String>,
    opsets: HashMap<String, i64>,
}

impl Model {
    /// Load ONNX model from a file path using memory mapping (`mmap`).
    ///
    /// The parent directory of `path` is used to resolve external data files.
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

        Self::load_from_bytes_with_dir(bytes, model_dir)
    }

    /// Load ONNX model from byte buffer (e.g., `Vec<u8>`, `Bytes`, `&'static [u8]`).
    ///
    /// Use [`Model::load_from_bytes_with_path`] if the model references external data files.
    pub fn load_from_bytes(data: impl Into<Bytes>) -> Result<Self, Error> {
        Self::load_from_bytes_with_dir(data.into(), None)
    }

    /// Load ONNX model from byte buffer (e.g., `Vec<u8>`, `Bytes`, `&'static [u8]`)
    /// with a directory path for resolving external data files.
    ///
    /// Note: `path` must be the folder/directory containing the external data files, not a file path.
    /// Use [`Model::load_from_bytes`] if the model contains no external data files.
    pub fn load_from_bytes_with_path<P: AsRef<Path>>(
        data: impl Into<Bytes>,
        path: P,
    ) -> Result<Self, Error> {
        Self::load_from_bytes_with_dir(data.into(), Some(path.as_ref().to_path_buf()))
    }

    /// Load ONNX model from `Bytes` buffer with optional model directory for external data
    fn load_from_bytes_with_dir(data: Bytes, model_dir: Option<PathBuf>) -> Result<Self, Error> {
        let model = ModelProto::decode(data)?;
        let graph = model.graph.ok_or(Error::MissingField("model graph"))?;

        // Create external data loader if model directory is available
        // Tensors keep the loader alive via Arc as long as they need it
        let external_data_loader = model_dir.map(|dir| Arc::new(ExternalDataLoader::new(dir)));

        let metadata: HashMap<String, String> = model
            .metadata_props
            .into_iter()
            .filter_map(|prop| prop.key.zip(prop.value))
            .collect();

        let opsets: HashMap<String, i64> = model
            .opset_import
            .into_iter()
            .map(|opset| {
                let version = opset.version.ok_or(Error::MissingField("opset version"))?;
                Ok((opset.domain.unwrap_or_default(), version))
            })
            .collect::<Result<_, Error>>()?;

        let ir_version = model
            .ir_version
            .ok_or(Error::MissingField("model ir_version"))?;

        Ok(Model {
            graph: proto_adapter::graph_from_proto(graph, external_data_loader.as_ref())?,
            ir_version,
            producer_name: model.producer_name,
            producer_version: model.producer_version,
            domain: model.domain,
            model_version: model.model_version,
            doc_string: model.doc_string,
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
    pub fn producer_name(&self) -> Option<&str> {
        self.producer_name.as_deref()
    }

    /// Get producer version
    pub fn producer_version(&self) -> Option<&str> {
        self.producer_version.as_deref()
    }

    /// Get model domain
    pub fn domain(&self) -> Option<&str> {
        self.domain.as_deref()
    }

    /// Get model version
    pub fn model_version(&self) -> Option<i64> {
        self.model_version
    }

    /// Get documentation string
    pub fn doc_string(&self) -> Option<&str> {
        self.doc_string.as_deref()
    }

    /// Get custom metadata properties
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Get operator set imports (domain -> version)
    pub fn opsets(&self) -> &HashMap<String, i64> {
        &self.opsets
    }

    /// Get the imported opset version for the default ONNX domain ("")
    pub fn default_opset_version(&self) -> Option<i64> {
        self.opsets.get("").copied()
    }
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ONNX Model Information:")?;
        writeln!(f, "IR Version: {}", self.ir_version)?;
        writeln!(f, "Producer Name: {:?}", self.producer_name)?;
        writeln!(f, "Producer Version: {:?}", self.producer_version)?;
        writeln!(f, "Domain: {:?}", self.domain)?;
        writeln!(f, "Model Version: {:?}", self.model_version)?;
        writeln!(f, "Description: {:?}", self.doc_string)?;
        writeln!(f, "Metadata: {:?}", self.metadata)?;
        writeln!(f, "Opsets: {:?}", self.opsets)?;

        writeln!(f, "\nRoot Graph & Subgraph Details:")?;
        write!(f, "{}", self.graph)
    }
}
