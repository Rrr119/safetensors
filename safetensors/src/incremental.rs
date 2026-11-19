//! Incremental loading of safetensors files.
//!
//! [`SafeTensors`](crate::SafeTensors) borrows a complete byte buffer. That
//! is ideal when the caller already has an mmap, but it forces callers to
//! materialise the whole file before they can ask for a single tensor.
//! [`IncrementalSafeTensors`] instead keeps the file on disk, parses only
//! the header, and reads tensor payloads one at a time.
//!
//! The on-disk format is unchanged. Files written by [`serialize`](crate::serialize)
//! or [`serialize_to_file`](crate::serialize_to_file) can be read by this
//! loader, and files read here remain standard safetensors files.
//!
//! ```no_run
//! use safetensors::incremental::{Backend, IncrementalSafeTensors};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let file = IncrementalSafeTensors::open("model.safetensors")?;
//! println!("{} tensors", file.len());
//!
//! // Reads only this tensor's bytes.
//! let tensor = file.tensor("model.embed_tokens.weight")?;
//! println!("{:?} {:?}", tensor.dtype(), tensor.shape());
//! # Ok(())
//! # }
//! ```

use crate::lib::{Cow, String, ToString, Vec};
use crate::tensor::{Dtype, Metadata, SafeTensorError, SafeTensors, TensorInfo, TensorView, View};
use std::fs::File;
use std::io;
#[cfg(not(any(unix, windows)))]
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

#[cfg(feature = "mmap")]
use memmap2::{Mmap, MmapOptions};

/// Where tensor bytes are read from.
///
/// `Auto` picks [`Backend::Mmap`] when the `mmap` feature is enabled and
/// [`Backend::Pread`] otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Backend {
    /// Let the loader choose the platform default.
    #[default]
    Auto,
    /// Memory-map the file and copy tensor ranges out of the mapping.
    Mmap,
    /// Read tensor ranges with positional reads (`pread` / `seek_read`).
    ///
    /// Use this on unified-memory machines where mapping the whole
    /// checkpoint would keep a second copy of the model resident.
    Pread,
}

/// Alias for [`Backend`].
pub type LoaderBackend = Backend;

/// A tensor loaded with its bytes owned by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Alias for [`OwnedTensor`].
pub type TensorData = OwnedTensor;

impl OwnedTensor {
    /// Creates an owned tensor from its parts.
    pub fn new(dtype: Dtype, shape: Vec<usize>, data: Vec<u8>) -> Result<Self, SafeTensorError> {
        crate::tensor::TensorView::new(dtype, shape.clone(), &data)?;
        Ok(Self { dtype, shape, data })
    }

    /// The tensor dtype.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// The tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The tensor bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Consumes the tensor and returns its bytes.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// Consumes the tensor and returns its dtype, shape and bytes.
    pub fn into_parts(self) -> (Dtype, Vec<usize>, Vec<u8>) {
        (self.dtype, self.shape, self.data)
    }

    /// The number of bytes in the tensor.
    pub fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl View for OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        self.data.as_slice().into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl View for &OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        self.data.as_slice().into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug)]
enum Source {
    #[cfg(feature = "mmap")]
    Mmap(Mmap),
    Pread(Mutex<File>),
}

/// An incrementally loaded safetensors file.
///
/// Opening the file reads only the little-endian header length and the
/// JSON header. Tensor data is read lazily by [`Self::tensor`],
/// [`Self::read_tensor`] or [`Self::read_tensor_into`].
#[derive(Debug)]
pub struct IncrementalSafeTensors {
    metadata: Metadata,
    source: Source,
    data_offset: u64,
    total_len: usize,
}

/// Alias for [`IncrementalSafeTensors`].
pub type SafeTensorsFile = IncrementalSafeTensors;

impl IncrementalSafeTensors {
    /// Opens a file using [`Backend::Auto`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SafeTensorError> {
        Self::open_with_backend(path, Backend::Auto)
    }

    /// Opens a file with an explicit storage backend.
    pub fn open_with_backend(
        path: impl AsRef<Path>,
        backend: Backend,
    ) -> Result<Self, SafeTensorError> {
        let file = File::open(path)?;
        Self::from_file_with_backend(file, backend)
    }

    /// Opens a file with the `mmap` backend.
    pub fn open_mmap(path: impl AsRef<Path>) -> Result<Self, SafeTensorError> {
        Self::open_with_backend(path, Backend::Mmap)
    }

    /// Opens a file with the positional-read backend.
    pub fn open_pread(path: impl AsRef<Path>) -> Result<Self, SafeTensorError> {
        Self::open_with_backend(path, Backend::Pread)
    }

    /// Builds a loader from an already-open file, using [`Backend::Auto`].
    pub fn from_file(file: File) -> Result<Self, SafeTensorError> {
        Self::from_file_with_backend(file, Backend::Auto)
    }

    /// Builds a loader from an already-open file.
    ///
    /// The file is read from its current position only for the header;
    /// tensor reads always use explicit absolute offsets.
    pub fn from_file_with_backend(
        mut file: File,
        backend: Backend,
    ) -> Result<Self, SafeTensorError> {
        let total_len: usize = file
            .metadata()?
            .len()
            .try_into()
            .map_err(|_| SafeTensorError::HeaderTooLarge)?;
        let (header_len, metadata) = SafeTensors::read_metadata_from_reader(&mut file, total_len)?;
        let data_offset = crate::tensor::N_LEN
            .checked_add(header_len)
            .ok_or(SafeTensorError::InvalidHeaderLength)? as u64;

        let backend = match backend {
            Backend::Auto => {
                #[cfg(feature = "mmap")]
                {
                    Backend::Mmap
                }
                #[cfg(not(feature = "mmap"))]
                {
                    Backend::Pread
                }
            }
            other => other,
        };

        let source = match backend {
            Backend::Mmap => {
                #[cfg(feature = "mmap")]
                {
                    // SAFETY: the mapping is only exposed through immutable
                    // slices, and the `Mmap` owns the mapping for as long as
                    // this loader exists.
                    Source::Mmap(unsafe { MmapOptions::new().map(&file)? })
                }
                #[cfg(not(feature = "mmap"))]
                {
                    return Err(SafeTensorError::IoError(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "the `mmap` feature is disabled",
                    )));
                }
            }
            Backend::Pread => Source::Pread(Mutex::new(file)),
            Backend::Auto => unreachable!("Auto is resolved above"),
        };

        Ok(Self {
            metadata,
            source,
            data_offset,
            total_len,
        })
    }

    /// The parsed header.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Header information for one tensor.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.metadata.info(name)
    }

    /// Alias for [`Self::info`], matching the Python binding's naming.
    pub fn get_tensor_meta(&self, name: &str) -> Option<&TensorInfo> {
        self.info(name)
    }

    /// Tensor names.
    pub fn names(&self) -> Vec<&str> {
        self.metadata.names()
    }

    /// Tensor names ordered by their data offset.
    pub fn offset_keys(&self) -> Vec<String> {
        self.metadata.offset_keys()
    }

    /// Number of tensors in the file.
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Whether the file contains no tensors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total size of the tensor payload.
    pub fn data_len(&self) -> usize {
        self.metadata.data_len()
    }

    /// Full file length in bytes.
    pub fn file_len(&self) -> usize {
        self.total_len
    }

    /// The backend selected for this loader.
    pub fn backend(&self) -> Backend {
        match &self.source {
            #[cfg(feature = "mmap")]
            Source::Mmap(_) => Backend::Mmap,
            Source::Pread(_) => Backend::Pread,
        }
    }

    /// The complete mapping, when the loader uses the `mmap` backend.
    #[cfg(feature = "mmap")]
    pub fn mmap(&self) -> Option<&[u8]> {
        match &self.source {
            Source::Mmap(mmap) => Some(mmap),
            Source::Pread(_) => None,
        }
    }

    /// The tensor's bytes, borrowed from the mapping when possible.
    ///
    /// With the `mmap` backend this is a borrowed slice and performs no
    /// copy. With the `pread` backend the bytes are allocated and read.
    pub fn tensor_slice(&self, name: &str) -> Result<Cow<'_, [u8]>, SafeTensorError> {
        let info = self.info_or_err(name)?;
        let (begin, end) = info.data_offsets;
        self.check_range(name, info, begin, end)?;

        match &self.source {
            #[cfg(feature = "mmap")]
            Source::Mmap(mmap) => {
                let start = self.data_offset as usize + begin;
                let stop = start + (end - begin);
                Ok(Cow::Borrowed(&mmap[start..stop]))
            }
            Source::Pread(file) => {
                let mut data = vec![0u8; end - begin];
                read_file(file, &mut data, self.data_offset + begin as u64)?;
                Ok(Cow::Owned(data))
            }
        }
    }

    /// Borrows one tensor directly from the mapping.
    ///
    /// Returns `Ok(None)` for the `pread` backend, where no mapping exists.
    /// With the `mmap` backend this is zero-copy.
    #[cfg(feature = "mmap")]
    pub fn tensor_view(&self, name: &str) -> Result<Option<TensorView<'_>>, SafeTensorError> {
        let info = self.info_or_err(name)?;
        let (begin, end) = info.data_offsets;
        self.check_range(name, info, begin, end)?;

        match &self.source {
            Source::Mmap(mmap) => {
                let start = self.data_offset as usize + begin;
                let data = &mmap[start..start + (end - begin)];
                Ok(Some(TensorView::new(info.dtype, info.shape.clone(), data)?))
            }
            Source::Pread(_) => Ok(None),
        }
    }

    /// Reads one tensor's bytes into `dst`.
    ///
    /// `dst` may be larger than the tensor; only the tensor's bytes are
    /// written and the number of bytes written is returned.
    pub fn read_tensor_into(&self, name: &str, dst: &mut [u8]) -> Result<usize, SafeTensorError> {
        let info = self.info_or_err(name)?;
        let (begin, end) = info.data_offsets;
        self.check_range(name, info, begin, end)?;
        let nbytes = end - begin;
        if dst.len() < nbytes {
            return Err(SafeTensorError::InvalidTensorView(
                info.dtype,
                info.shape.clone(),
                dst.len(),
            ));
        }

        let dst = &mut dst[..nbytes];
        match &self.source {
            #[cfg(feature = "mmap")]
            Source::Mmap(mmap) => {
                let start = self.data_offset as usize + begin;
                dst.copy_from_slice(&mmap[start..start + nbytes]);
            }
            Source::Pread(file) => {
                read_file(file, dst, self.data_offset + begin as u64)?;
            }
        }
        Ok(nbytes)
    }

    /// Reads one tensor's bytes.
    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>, SafeTensorError> {
        let info = self.info_or_err(name)?;
        let mut data = vec![0u8; info.data_offsets.1 - info.data_offsets.0];
        self.read_tensor_into(name, &mut data)?;
        Ok(data)
    }

    /// Reads a byte range inside one tensor.
    ///
    /// This is useful for chunking a very large tensor without loading
    /// the whole tensor: `range` is relative to the tensor's payload.
    pub fn read_tensor_range(
        &self,
        name: &str,
        range: core::ops::Range<usize>,
    ) -> Result<Vec<u8>, SafeTensorError> {
        let info = self.info_or_err(name)?;
        let (begin, end) = info.data_offsets;
        self.check_range(name, info, begin, end)?;
        let nbytes = end - begin;
        if range.start > range.end || range.end > nbytes {
            return Err(SafeTensorError::InvalidOffset(name.to_string()));
        }

        let len = range.end - range.start;
        let mut data = vec![0u8; len];
        let absolute = self.data_offset + (begin + range.start) as u64;
        match &self.source {
            #[cfg(feature = "mmap")]
            Source::Mmap(mmap) => {
                let start = absolute as usize;
                data.copy_from_slice(&mmap[start..start + len]);
            }
            Source::Pread(file) => read_file(file, &mut data, absolute)?,
        }
        Ok(data)
    }

    /// Loads one tensor into an owned [`OwnedTensor`].
    pub fn tensor(&self, name: &str) -> Result<OwnedTensor, SafeTensorError> {
        let info = self.info_or_err(name)?;
        let dtype = info.dtype;
        let shape = info.shape.clone();
        let data = self.read_tensor(name)?;
        OwnedTensor::new(dtype, shape, data)
    }

    /// Alias for [`Self::tensor`].
    pub fn load_tensor(&self, name: &str) -> Result<OwnedTensor, SafeTensorError> {
        self.tensor(name)
    }

    /// Alias for [`Self::tensor`], matching the Python binding's naming.
    pub fn get_tensor(&self, name: &str) -> Result<OwnedTensor, SafeTensorError> {
        self.tensor(name)
    }

    /// Loads every tensor, one at a time.
    pub fn tensors(&self) -> Result<Vec<(String, OwnedTensor)>, SafeTensorError> {
        self.iter_tensors().collect()
    }

    /// Iterates over tensors, loading each one when it is yielded.
    pub fn iter_tensors(
        &self,
    ) -> impl Iterator<Item = Result<(String, OwnedTensor), SafeTensorError>> + '_ {
        self.offset_keys()
            .into_iter()
            .map(move |name| Ok((name.clone(), self.tensor(&name)?)))
    }

    fn info_or_err(&self, name: &str) -> Result<&TensorInfo, SafeTensorError> {
        self.metadata
            .info(name)
            .ok_or_else(|| SafeTensorError::TensorNotFound(name.to_string()))
    }

    fn check_range(
        &self,
        name: &str,
        info: &TensorInfo,
        begin: usize,
        end: usize,
    ) -> Result<(), SafeTensorError> {
        let data_len = self.metadata.data_len();
        if begin > end || end > data_len {
            return Err(SafeTensorError::InvalidOffset(name.to_string()));
        }
        if end - begin != info.data_offsets.1 - info.data_offsets.0 {
            return Err(SafeTensorError::TensorInvalidInfo);
        }
        Ok(())
    }
}

fn read_file(file: &Mutex<File>, buf: &mut [u8], offset: u64) -> Result<(), SafeTensorError> {
    let file = file
        .lock()
        .map_err(|_| SafeTensorError::IoError(io::Error::other("file lock poisoned")))?;
    read_exact_at(&file, buf, offset)?;
    Ok(())
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;

    while !buf.is_empty() {
        match file.read_at(buf, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;

    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{serialize, serialize_to_file, SafeTensors, TensorView};
    use std::collections::HashMap;
    use std::io::Cursor;

    fn fixture() -> (tempfile::NamedTempFile, Vec<u8>) {
        let first: Vec<u8> = (0..64u8).collect();
        let second: Vec<u8> = (100..164u8).collect();
        let empty: Vec<u8> = Vec::new();

        let tensors = [
            (
                "a".to_string(),
                TensorView::new(Dtype::U8, vec![8, 8], &first).unwrap(),
            ),
            (
                "b".to_string(),
                TensorView::new(Dtype::U8, vec![4, 4, 4], &second).unwrap(),
            ),
            (
                "empty".to_string(),
                TensorView::new(Dtype::U8, vec![0, 3], &empty).unwrap(),
            ),
        ];

        let file = tempfile::NamedTempFile::new().unwrap();
        serialize_to_file(
            tensors.iter().map(|(name, view)| (name, view)),
            Some(HashMap::from([("format".to_string(), "test".to_string())])),
            file.path(),
        )
        .unwrap();

        let second_bytes = second.clone();
        (file, second_bytes)
    }

    #[test]
    fn reads_only_the_header_from_a_reader() {
        let first = vec![1u8; 16];
        let view = TensorView::new(Dtype::U8, vec![16], &first).unwrap();
        let bytes = serialize([("t", view)], None).unwrap();

        struct CountingReader {
            cursor: Cursor<Vec<u8>>,
            read: usize,
        }
        impl std::io::Read for CountingReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let n = self.cursor.read(buf)?;
                self.read += n;
                Ok(n)
            }
        }

        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let mut reader = CountingReader {
            cursor: Cursor::new(bytes.clone()),
            read: 0,
        };
        let (n, metadata) =
            SafeTensors::read_metadata_from_reader(&mut reader, bytes.len()).unwrap();
        assert_eq!(n, header_len);
        assert_eq!(reader.read, 8 + header_len);
        assert!(metadata.info("t").is_some());
    }

    #[test]
    fn pread_and_mmap_read_the_same_tensors() {
        let (file, second) = fixture();

        let pread = IncrementalSafeTensors::open_pread(file.path()).unwrap();
        assert_eq!(pread.backend(), Backend::Pread);
        assert_eq!(pread.len(), 3);
        assert_eq!(pread.tensor("b").unwrap().data(), second.as_slice());

        let mut larger = vec![0xAB; second.len() + 8];
        assert_eq!(
            pread.read_tensor_into("b", &mut larger).unwrap(),
            second.len()
        );
        assert_eq!(&larger[..second.len()], second.as_slice());
        assert_eq!(&larger[second.len()..], &[0xAB; 8]);

        assert_eq!(
            pread.read_tensor_range("b", 3..19).unwrap(),
            second[3..19].to_vec()
        );
        assert_eq!(pread.read_tensor("empty").unwrap(), Vec::<u8>::new());
        assert_eq!(pread.tensors().unwrap().len(), 3);

        let loaded = pread.tensor("b").unwrap();
        let roundtrip = serialize([("b", loaded)], None).unwrap();
        let decoded = SafeTensors::deserialize(&roundtrip).unwrap();
        assert_eq!(decoded.tensor("b").unwrap().data(), second.as_slice());

        #[cfg(feature = "mmap")]
        {
            let mmap = IncrementalSafeTensors::open_mmap(file.path()).unwrap();
            assert_eq!(mmap.backend(), Backend::Mmap);
            assert_eq!(mmap.tensor("b").unwrap().data(), second.as_slice());
            assert_eq!(mmap.tensor("b").unwrap(), pread.tensor("b").unwrap());
            assert!(mmap.mmap().is_some());
            let view = mmap.tensor_view("b").unwrap().unwrap();
            assert_eq!(view.data(), second.as_slice());
            let mapping = mmap.mmap().unwrap();
            let start = view.data().as_ptr() as usize;
            assert!(start >= mapping.as_ptr() as usize);
            assert!(start + view.data().len() <= mapping.as_ptr() as usize + mapping.len());
        }
    }

    #[test]
    fn missing_tensor_errors() {
        let (file, _) = fixture();
        let loader = IncrementalSafeTensors::open_pread(file.path()).unwrap();
        assert!(matches!(
            loader.tensor("missing"),
            Err(SafeTensorError::TensorNotFound(_))
        ));
    }

    #[test]
    fn incomplete_reader_is_rejected() {
        let first = vec![1u8; 16];
        let view = TensorView::new(Dtype::U8, vec![16], &first).unwrap();
        let bytes = serialize([("t", view)], None).unwrap();

        let err = SafeTensors::read_metadata_from_reader(Cursor::new(&bytes[..8]), bytes.len())
            .unwrap_err();
        assert!(matches!(err, SafeTensorError::InvalidHeaderLength));
    }
}
