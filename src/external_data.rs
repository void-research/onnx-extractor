use memmap2::Mmap;
use prost::bytes::Bytes;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::{Error, StringStringEntryProto};

#[derive(Debug, Clone)]
pub struct ExternalDataInfo {
    location: String,
    offset: Option<u64>,
    length: Option<u64>,
    loader: Arc<ExternalDataLoader>,
}

impl ExternalDataInfo {
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

        let location = location.ok_or(Error::MissingField("external data location"))?;

        Ok(ExternalDataInfo {
            location,
            offset,
            length,
            loader,
        })
    }

    pub fn load_data(&self) -> Result<Bytes, Error> {
        self.loader.load_data(self)
    }
}

pub struct ExternalDataLoader {
    model_dir: PathBuf,
    cache: RwLock<HashMap<String, Bytes>>,
}

impl ExternalDataLoader {
    pub fn new(model_dir: PathBuf) -> Self {
        ExternalDataLoader {
            model_dir,
            cache: RwLock::new(HashMap::new()),
        }
    }

    // This method lazily loads the entire external file into the cache on first access,
    // then returns a slice of the cached data based on offset and length.
    // It uses a slower hold lock while loading style to stop multiple threads from
    // loading the same file into memory if both race past read.
    fn load_data(&self, info: &ExternalDataInfo) -> Result<Bytes, Error> {
        {
            let cache = self.cache.read()?;
            if let Some(cached_data) = cache.get(&info.location) {
                return Self::slice_data(cached_data, info);
            }
        }

        let mut cache = self.cache.write()?;

        if let Some(cached_data) = cache.get(&info.location) {
            return Self::slice_data(cached_data, info);
        }

        let file_path = self.model_dir.join(&info.location);
        let file = File::open(file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let full_file_data = Bytes::from_owner(mmap);
        let data_slice = Self::slice_data(&full_file_data, info)?;

        cache.insert(info.location.clone(), full_file_data);

        Ok(data_slice)
    }

    fn slice_data(data: &Bytes, info: &ExternalDataInfo) -> Result<Bytes, Error> {
        let start = info.offset.unwrap_or(0) as usize;
        let end = info
            .length
            .map_or(data.len(), |len| start.saturating_add(len as usize));

        if start > end || end > data.len() {
            return Err(Error::ExternalDataOutOfBounds {
                start,
                end,
                file_size: data.len(),
            });
        }

        Ok(data.slice(start..end))
    }
}

impl std::fmt::Debug for ExternalDataLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cached_count = self.cache.read().map(|c| c.len()).unwrap_or_default();
        f.debug_struct("ExternalDataLoader")
            .field("model_dir", &self.model_dir)
            .field("cached_files", &cached_count)
            .finish()
    }
}
