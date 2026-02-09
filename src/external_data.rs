use memmap2::MmapOptions;
use prost::bytes::Bytes;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::{Error, StringStringEntryProto};

/// Metadata for external tensor data
#[derive(Debug, Clone)]
pub(crate) struct ExternalDataInfo {
    pub location: String,
    pub offset: Option<u64>,
    pub length: Option<u64>,
    pub loader: Rc<ExternalDataLoader>,
}

impl ExternalDataInfo {
    /// Parse external data info from key-value pairs
    pub fn from_key_value_pairs(
        pairs: &[StringStringEntryProto],
        loader: Rc<ExternalDataLoader>,
    ) -> Result<Self, Error> {
        let mut location: Option<String> = None;
        let mut offset: Option<u64> = None;
        let mut length: Option<u64> = None;

        for pair in pairs {
            let key = pair.key.as_deref().unwrap_or("");
            let value = pair.value.as_deref().unwrap_or("");

            match key {
                "location" => location = Some(value.to_string()),
                "offset" => {
                    offset = value.parse::<u64>().ok();
                }
                "length" => {
                    length = value.parse::<u64>().ok();
                }
                _ => {} // ignore unknown keys
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
    cache: RefCell<HashMap<String, Bytes>>,
}

impl ExternalDataLoader {
    /// Create a new external data loader for a given model directory
    pub(crate) fn new(model_dir: PathBuf) -> Self {
        ExternalDataLoader {
            model_dir,
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// Load tensor data from external file with optional offset and length
    ///
    /// This method lazily loads the entire external file into the cache on first access,
    /// then returns a slice of the cached data based on offset and length.
    pub(crate) fn load_data(&self, info: &ExternalDataInfo) -> Result<Bytes, Error> {
        let file_path = self.model_dir.join(&info.location);

        {
            let cache = self.cache.borrow();
            if let Some(cached_data) = cache.get(&info.location) {
                // File is cached, return the requested slice
                return self.slice_data(cached_data, info);
            }
        }

        // File not cached
        let file_data = self.load_file(&file_path)?;

        let slice = self.slice_data(&file_data, info)?;

        // Cache the entire file
        let mut cache = self.cache.borrow_mut();
        cache.insert(info.location.clone(), file_data);

        Ok(slice)
    }

    /// Memory-map entire file as Bytes
    pub(crate) fn load_file(&self, path: &Path) -> Result<Bytes, Error> {
        let file = File::open(path).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!(
                    "Failed to open external data file '{}': {}",
                    path.display(),
                    e
                ),
            ))
        })?;

        let mmap = unsafe {
            MmapOptions::new().map_copy_read_only(&file).map_err(|e| {
                Error::Io(std::io::Error::new(
                    e.kind(),
                    format!(
                        "Failed to mmap external data file '{}': {}",
                        path.display(),
                        e
                    ),
                ))
            })?
        };

        Ok(Bytes::from_owner(mmap))
    }

    /// Extract a slice of data based on offset and length
    pub(crate) fn slice_data(&self, data: &Bytes, info: &ExternalDataInfo) -> Result<Bytes, Error> {
        let start = info.offset.unwrap_or(0) as usize;
        let end = if let Some(len) = info.length {
            start.saturating_add(len as usize)
        } else {
            data.len()
        };

        if start > data.len() {
            return Err(Error::InvalidModel(format!(
                "External data offset {} exceeds file size {}",
                start,
                data.len()
            )));
        }

        if end > data.len() {
            return Err(Error::InvalidModel(format!(
                "External data range {}..{} exceeds file size {}",
                start,
                end,
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
            .field("cached_files", &self.cache.borrow().len())
            .finish()
    }
}
