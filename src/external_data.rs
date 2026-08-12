use memmap2::Mmap;
use prost::bytes::Bytes;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::{Error, StringStringEntryProto};

/// Metadata for external tensor data
#[derive(Debug, Clone)]
pub(crate) struct ExternalDataInfo {
    pub location: String,
    pub offset: Option<u64>,
    pub length: Option<u64>,
    pub loader: Arc<ExternalDataLoader>,
}

impl ExternalDataInfo {
    /// Parse external data info from key-value pairs
    pub fn from_key_value_pairs(
        pairs: Vec<StringStringEntryProto>,
        loader: Arc<ExternalDataLoader>,
    ) -> Result<Self, Error> {
        let mut location: Option<String> = None;
        let mut offset: Option<u64> = None;
        let mut length: Option<u64> = None;

        for pair in pairs {
            if let (Some(k), Some(v)) = (pair.key.as_deref(), pair.value) {
                match k {
                    "location" => location = Some(v),
                    "offset" => offset = Some(v.parse()?),
                    "length" => length = Some(v.parse()?),
                    _ => {}
                }
            }
        }

        let location = location.ok_or_else(|| {
            Error::InvalidModel("External data missing required 'location' field".to_string())
        })?;

        Ok(ExternalDataInfo {
            location,
            offset,
            length,
            loader,
        })
    }

    /// Load the external data using the stored loader
    pub fn load_data(&self) -> Result<Bytes, Error> {
        self.loader.load_data(self)
    }
}

/// Manages lazy loading and caching of external tensor data files
pub(crate) struct ExternalDataLoader {
    model_dir: PathBuf,
    cache: RwLock<HashMap<String, Bytes>>,
}

impl ExternalDataLoader {
    /// Create a new external data loader for a given model directory
    pub(crate) fn new(model_dir: PathBuf) -> Self {
        ExternalDataLoader {
            model_dir,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Load tensor data from external file with optional offset and length
    ///
    /// This method lazily loads the entire external file into the cache on first access,
    /// then returns a slice of the cached data based on offset and length.
    pub(crate) fn load_data(&self, info: &ExternalDataInfo) -> Result<Bytes, Error> {
        {
            let cache = self.cache.read().unwrap();
            if let Some(cached_data) = cache.get(&info.location) {
                return self.slice_data(cached_data, info);
            }
        }

        let mut cache = self.cache.write().unwrap();

        if let Some(cached_data) = cache.get(&info.location) {
            return self.slice_data(cached_data, info);
        }

        let file_path = self.model_dir.join(&info.location);
        let file_data = self.load_file(&file_path)?;
        let slice = self.slice_data(&file_data, info)?;

        cache.insert(info.location.clone(), file_data);

        Ok(slice)
    }

    /// Memory-map entire file as Bytes
    pub(crate) fn load_file(&self, path: &Path) -> Result<Bytes, Error> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Bytes::from_owner(mmap))
    }

    /// Extract a slice of data based on offset and length
    pub(crate) fn slice_data(&self, data: &Bytes, info: &ExternalDataInfo) -> Result<Bytes, Error> {
        let start = info.offset.unwrap_or(0) as usize;
        let end = info
            .length
            .map_or(data.len(), |len| start.saturating_add(len as usize));

        if start > data.len() {
            return Err(Error::InvalidModel(format!(
                "External data offset {start} exceeds file size {}",
                data.len()
            )));
        }

        if end > data.len() {
            return Err(Error::InvalidModel(format!(
                "External data range {start}..{end} exceeds file size {}",
                data.len()
            )));
        }

        Ok(data.slice(start..end))
    }
}

impl std::fmt::Debug for ExternalDataLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalDataLoader")
            .field("model_dir", &self.model_dir)
            .field("cached_files", &self.cache.read().unwrap().len())
            .finish()
    }
}
