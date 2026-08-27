//! Functions that enable extracting or streaming a Conda package for objects
//! that implement the [`tokio::io::AsyncRead`] trait.

use std::path::Path;

use async_compression::tokio::bufread::BzDecoder;
use async_spooled_tempfile::SpooledTempFile;
use async_zip::base::read::stream::ZipFileReader;
#[cfg(feature = "reqwest")]
use futures_util::StreamExt;
use tokio::io::{AsyncRead, AsyncSeekExt};
use tokio_util::compat::{FuturesAsyncReadCompatExt, TokioAsyncReadCompatExt};

use crate::{ExtractError, ExtractResult, read::SizeCountingReader};

use super::shared::{DEFAULT_BUF_SIZE, extract_tar_zst_entry, unpack_tar_archive};

/// Extracts the contents a `.tar.bz2` package archive using fully async implementation.
pub async fn extract_tar_bz2(
    reader: impl AsyncRead + Send + Unpin + 'static,
    destination: &Path,
) -> Result<ExtractResult, ExtractError> {
    // Ensure the destination directory exists
    tokio::fs::create_dir_all(destination)
        .await
        .map_err(ExtractError::CouldNotCreateDestination)?;

    // Clone destination for the async block
    let destination = destination.to_owned();

    // Wrap the reading in additional readers that will compute the hashes while extracting
    let sha256_reader = rattler_digest::HashingReader::<_, rattler_digest::Sha256>::new(reader);
    let mut md5_reader =
        rattler_digest::HashingReader::<_, rattler_digest::Md5>::new(sha256_reader);
    let mut size_reader = SizeCountingReader::new(&mut md5_reader);

    // Create a buffered reader for better performance
    let buf_reader = tokio::io::BufReader::with_capacity(DEFAULT_BUF_SIZE, &mut size_reader);

    // Decompress bzip2 asynchronously
    let decoder = BzDecoder::new(buf_reader);

    // Build archive with optimized settings for faster extraction:
    // - Skip automatic mtime preservation (we set mtimes manually with safe clamping)
    // - Skip automatic permission handling (we'll set executable bits manually)
    // - Skip extended attributes for better performance
    let archive = tokio_tar::ArchiveBuilder::new(decoder)
        .set_preserve_mtime(false)
        .set_preserve_permissions(false)
        .set_unpack_xattrs(false)
        // We need this setting otherwise some packages in conda-forge will
        // not extract. However, we are checking much better in rattler-build and hopefully
        // one day can remove this.
        .set_allow_external_symlinks(true)
        .build();

    // Unpack entries manually, preserving only executable bits on Unix
    unpack_tar_archive(archive, &destination).await?;

    // Read the file to the end to make sure the hash is properly computed
    tokio::io::copy(&mut size_reader, &mut tokio::io::sink())
        .await
        .map_err(ExtractError::IoError)?;

    // Get the size and hashes
    let (_, total_size) = size_reader.finalize();
    let (sha256_reader, md5) = md5_reader.finalize();
    let (_, sha256) = sha256_reader.finalize();

    // Validate that we actually read some data from the stream.
    // If total_size is 0, it likely means the stream was truncated or the bzip2
    // decompressor silently failed without detecting an incomplete stream.
    if total_size == 0 {
        return Err(ExtractError::IoError(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "no data was read from the package stream - the stream may have been truncated",
        )));
    }

    Ok(ExtractResult {
        sha256,
        md5,
        total_size,
    })
}

/// Extracts the contents of a `.conda` package archive using fully async implementation.
/// This will perform on-the-fly decompression by streaming the reader.
pub async fn extract_conda(
    reader: impl AsyncRead + Send + Unpin + 'static,
    destination: &Path,
) -> Result<ExtractResult, ExtractError> {
    // Ensure the destination directory exists
    tokio::fs::create_dir_all(destination)
        .await
        .map_err(ExtractError::CouldNotCreateDestination)?;

    // Clone destination for the async block
    let destination = destination.to_owned();

    // Wrap the reading in additional readers that will compute the hashes while extracting
    let sha256_reader = rattler_digest::HashingReader::<_, rattler_digest::Sha256>::new(reader);
    let mut md5_reader =
        rattler_digest::HashingReader::<_, rattler_digest::Md5>::new(sha256_reader);
    let mut size_reader = SizeCountingReader::new(&mut md5_reader);

    // Convert to futures traits and create a buffered reader (async_zip uses futures traits)
    let compat_reader = (&mut size_reader).compat();
    let mut buf_reader = futures::io::BufReader::with_capacity(DEFAULT_BUF_SIZE, compat_reader);

    // Create a ZIP reader for streaming
    let mut zip_reader = ZipFileReader::new(&mut buf_reader);

    // Process each ZIP entry
    while let Some(mut entry) = zip_reader
        .next_with_entry()
        .await
        .map_err(|e| ExtractError::IoError(std::io::Error::other(e)))?
    {
        let filename = entry.reader().entry().filename().as_str().map_err(|e| {
            ExtractError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;

        // Only extract .tar.zst files
        if filename.ends_with(".tar.zst") {
            // Get a reader for the entry and convert from futures traits to tokio traits
            let mut compat_entry = entry.reader_mut().compat();
            extract_tar_zst_entry(&mut compat_entry, &destination).await?;
        }

        // Skip to the next entry (required by async_zip API)
        (.., zip_reader) = entry
            .skip()
            .await
            .map_err(|e| ExtractError::IoError(std::io::Error::other(e)))?;
    }

    // Read any remaining data to ensure hash is properly computed
    // Use futures copy since we're already in futures ecosystem
    futures::io::copy(&mut buf_reader, &mut futures::io::sink())
        .await
        .map_err(ExtractError::IoError)?;

    // Get the size and hashes
    let (_, total_size) = size_reader.finalize();
    let (sha256_reader, md5) = md5_reader.finalize();
    let (_, sha256) = sha256_reader.finalize();

    // Validate that we actually read some data from the stream
    if total_size == 0 {
        return Err(ExtractError::IoError(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "no data was read from the package stream - the stream may have been truncated",
        )));
    }

    Ok(ExtractResult {
        sha256,
        md5,
        total_size,
    })
}

/// Extracts the contents of a .conda package archive by fully reading the
/// stream and then decompressing. This is a fallback method for when streaming fails.
///
/// This implementation uses a `SpooledTempFile` (5MB in-memory threshold) to buffer
/// the package data, then uses the seek-based ZIP API for efficient extraction.
pub async fn extract_conda_via_buffering(
    reader: impl AsyncRead + Send + Unpin + 'static,
    destination: &Path,
) -> Result<ExtractResult, ExtractError> {
    // Delete destination first if it exists, as this method is usually used as a fallback
    if tokio::fs::try_exists(destination)
        .await
        .map_err(ExtractError::IoError)?
    {
        tokio::fs::remove_dir_all(destination)
            .await
            .map_err(ExtractError::CouldNotCreateDestination)?;
    }

    // Ensure the destination directory exists
    tokio::fs::create_dir_all(destination)
        .await
        .map_err(ExtractError::CouldNotCreateDestination)?;

    // Clone destination for the async block
    let destination = destination.to_owned();

    // Wrap the reading in additional readers that will compute the hashes while extracting
    let sha256_reader = rattler_digest::HashingReader::<_, rattler_digest::Sha256>::new(reader);
    let mut md5_reader =
        rattler_digest::HashingReader::<_, rattler_digest::Md5>::new(sha256_reader);
    let mut size_reader = SizeCountingReader::new(&mut md5_reader);

    // Create a SpooledTempFile (uses memory up to 5MB, then switches to disk)
    let mut spooled_file = SpooledTempFile::new(5 * 1024 * 1024);

    // Copy from reader to spooled file while computing hashes
    tokio::io::copy(&mut size_reader, &mut spooled_file)
        .await
        .map_err(ExtractError::IoError)?;

    // Get the size and hashes now that we've read everything
    let (_, total_size) = size_reader.finalize();
    let (sha256_reader, md5) = md5_reader.finalize();
    let (_, sha256) = sha256_reader.finalize();

    // Validate that we actually read some data from the stream
    if total_size == 0 {
        return Err(ExtractError::IoError(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "no data was read from the package stream - the stream may have been truncated",
        )));
    }

    // Rewind the spooled file to the beginning
    spooled_file.rewind().await.map_err(ExtractError::IoError)?;

    // Use the seek-based extraction (doesn't recompute hashes, we already have them)
    crate::tokio::async_seek::extract_conda(spooled_file, &destination).await?;

    Ok(ExtractResult {
        sha256,
        md5,
        total_size,
    })
}

/// Extracts the contents of a wheel (`.whl`) archive using a fully async,
/// streaming implementation. This performs on-the-fly extraction while the
/// archive is being read (e.g. while it's being downloaded), without
/// requiring the whole archive to be buffered first.
///
/// Each entry's original archive-relative destination path (i.e. the same
/// layout `unzip` would produce; see the module-level documentation of
/// [`rattler_conda_types::package::wheel`] for why no remapping is applied
/// during extraction) is written to as soon as its local file header is
/// encountered. Unix executable permission bits, however, are only stored in
/// the *central directory* at the end of the archive - not in the local
/// headers - so after streaming through every entry, this continues reading
/// forward into the central directory (mirroring what `uv` and `pip` do) to
/// recover and apply them.
///
/// Like [`extract_conda`], this can fail with
/// [`ExtractError::ZipError`]`(`[`zip::result::ZipError::UnsupportedArchive`]`(_))`
/// if the archive uses zip data descriptors in a way that's incompatible
/// with streaming; callers should fall back to
/// [`extract_wheel_via_buffering`] in that case.
pub async fn extract_wheel(
    reader: impl AsyncRead + Send + Unpin + 'static,
    destination: &Path,
) -> Result<ExtractResult, ExtractError> {
    // Ensure the destination directory exists
    tokio::fs::create_dir_all(destination)
        .await
        .map_err(ExtractError::CouldNotCreateDestination)?;

    // Clone destination for the async block
    let destination = destination.to_owned();

    // Wrap the reading in additional readers that will compute the hashes while extracting
    let sha256_reader = rattler_digest::HashingReader::<_, rattler_digest::Sha256>::new(reader);
    let mut md5_reader =
        rattler_digest::HashingReader::<_, rattler_digest::Md5>::new(sha256_reader);
    let mut size_reader = SizeCountingReader::new(&mut md5_reader);

    // Convert to futures traits and create a buffered reader (async_zip uses futures traits)
    let compat_reader = (&mut size_reader).compat();
    let mut buf_reader = futures::io::BufReader::with_capacity(DEFAULT_BUF_SIZE, compat_reader);

    // Create a ZIP reader for streaming
    let mut zip_reader = ZipFileReader::new(&mut buf_reader);

    // Records, for every non-directory entry we've written, its local-header
    // file offset -> destination path, so that we can look up and fix up
    // executable bits once we reach the matching central-directory entry.
    // Only needed on Unix, where executable bits are meaningful.
    #[cfg(unix)]
    let mut offset_to_path: std::collections::HashMap<u64, std::path::PathBuf> =
        std::collections::HashMap::new();

    let mut offset = 0u64;
    while let Some(mut entry) = zip_reader
        .next_with_entry()
        .await
        .map_err(|e| ExtractError::IoError(std::io::Error::other(e)))?
    {
        let zip_entry = entry.reader().entry();
        let filename = zip_entry.filename().as_str().map_err(|e| {
            ExtractError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;
        let is_dir = zip_entry.dir().unwrap_or(false);
        let file_offset = zip_entry.file_offset();

        let Some(relpath) = super::shared::sanitize_zip_entry_name(filename) else {
            tracing::warn!("skipping unsafe wheel entry path: {filename}");
            (.., zip_reader) = entry
                .skip()
                .await
                .map_err(|e| ExtractError::IoError(std::io::Error::other(e)))?;
            offset = zip_reader.offset();
            continue;
        };
        // Preserve the original archive-relative layout (i.e. the same
        // layout `unzip` would produce): the `site-packages`/
        // `python-scripts` remapping is applied later, at install time, not
        // during extraction. See the module-level documentation of
        // `rattler_conda_types::package::wheel` for why.
        let out_path = destination.join(&relpath);

        if is_dir {
            tokio::fs::create_dir_all(&out_path)
                .await
                .map_err(ExtractError::IoError)?;
        } else {
            if let Some(parent) = out_path.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(ExtractError::IoError)?;
            }

            let mut out_file = tokio::fs::File::create(&out_path)
                .await
                .map_err(ExtractError::IoError)?;
            let mut compat_entry = entry.reader_mut().compat();
            tokio::io::copy(&mut compat_entry, &mut out_file)
                .await
                .map_err(ExtractError::IoError)?;

            #[cfg(unix)]
            offset_to_path.insert(file_offset, out_path);
        }

        // Skip to the next entry (required by async_zip API)
        (.., zip_reader) = entry
            .skip()
            .await
            .map_err(|e| ExtractError::IoError(std::io::Error::other(e)))?;
        offset = zip_reader.offset();
    }

    // Central-directory pass (Unix only): recover executable bits, which
    // live in the central directory's external file attributes rather than
    // the local file headers we've just streamed through. The streaming
    // `ZipFileReader` stops right at the start of the central directory, so
    // we continue reading from there.
    #[cfg(unix)]
    {
        use async_zip::base::read::cd::{CentralDirectoryReader, Entry};

        let mut directory = CentralDirectoryReader::new(&mut buf_reader, offset);
        loop {
            match directory.next().await {
                Ok(Entry::CentralDirectoryEntry(entry)) => {
                    if let Some(path) = offset_to_path.get(&entry.file_offset()) {
                        super::shared::apply_executable_bit(entry.unix_permissions(), path).await?;
                    }
                }
                Ok(Entry::EndOfCentralDirectoryRecord { .. }) => break,
                Err(e) => {
                    // Best-effort: the archive has already been fully
                    // extracted above; failing to parse the central
                    // directory just means we lose executable-bit fixups.
                    tracing::warn!(
                        "failed to read wheel central directory for executable-bit fixup: {e}"
                    );
                    break;
                }
            }
        }
    }

    // Read any remaining data to ensure the hash is properly computed.
    futures::io::copy(&mut buf_reader, &mut futures::io::sink())
        .await
        .map_err(ExtractError::IoError)?;

    // Get the size and hashes
    let (_, total_size) = size_reader.finalize();
    let (sha256_reader, md5) = md5_reader.finalize();
    let (_, sha256) = sha256_reader.finalize();

    // Validate that we actually read some data from the stream
    if total_size == 0 {
        return Err(ExtractError::IoError(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "no data was read from the package stream - the stream may have been truncated",
        )));
    }

    Ok(ExtractResult {
        sha256,
        md5,
        total_size,
    })
}

/// Extracts the contents of a wheel (`.whl`) archive by fully reading the
/// stream and then extracting via the seek-based API. This is a fallback
/// method for when streaming (see [`extract_wheel`]) fails, e.g. due to zip
/// data descriptors that are incompatible with streaming decompression.
///
/// Like [`extract_conda_via_buffering`], this uses a [`SpooledTempFile`]
/// (5MB in-memory threshold, spilling to a real temporary file on disk
/// beyond that) rather than an unconditional in-memory buffer, so this
/// scales gracefully to the very large wheels shipped by some compiled
/// packages.
pub async fn extract_wheel_via_buffering(
    reader: impl AsyncRead + Send + Unpin + 'static,
    destination: &Path,
) -> Result<ExtractResult, ExtractError> {
    // Delete destination first if it exists, as this method is usually used as a fallback
    if tokio::fs::try_exists(destination)
        .await
        .map_err(ExtractError::IoError)?
    {
        tokio::fs::remove_dir_all(destination)
            .await
            .map_err(ExtractError::CouldNotCreateDestination)?;
    }

    // Ensure the destination directory exists
    tokio::fs::create_dir_all(destination)
        .await
        .map_err(ExtractError::CouldNotCreateDestination)?;

    // Clone destination for the async block
    let destination = destination.to_owned();

    // Wrap the reading in additional readers that will compute the hashes while
    // buffering the archive.
    let sha256_reader = rattler_digest::HashingReader::<_, rattler_digest::Sha256>::new(reader);
    let mut md5_reader =
        rattler_digest::HashingReader::<_, rattler_digest::Md5>::new(sha256_reader);
    let mut size_reader = SizeCountingReader::new(&mut md5_reader);

    // Create a SpooledTempFile (uses memory up to 5MB, then switches to disk)
    let mut spooled_file = SpooledTempFile::new(5 * 1024 * 1024);

    // Copy from reader to spooled file while computing hashes
    tokio::io::copy(&mut size_reader, &mut spooled_file)
        .await
        .map_err(ExtractError::IoError)?;

    // Get the size and hashes now that we've read everything
    let (_, total_size) = size_reader.finalize();
    let (sha256_reader, md5) = md5_reader.finalize();
    let (_, sha256) = sha256_reader.finalize();

    // Validate that we actually read some data from the stream
    if total_size == 0 {
        return Err(ExtractError::IoError(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "no data was read from the package stream - the stream may have been truncated",
        )));
    }

    // Rewind the spooled file to the beginning
    spooled_file.rewind().await.map_err(ExtractError::IoError)?;

    // Use the seek-based extraction (doesn't recompute hashes, we already have them)
    crate::tokio::async_seek::extract_wheel(spooled_file, &destination).await?;

    Ok(ExtractResult {
        sha256,
        md5,
        total_size,
    })
}

/// Async equivalent of [`crate::seek::get_file_from_archive`].
///
/// Iterates entries of a tar archive, returning the contents of the first
/// entry whose path matches `file_name`. Because the reader is streaming,
/// only the bytes up to (and including) the target entry are consumed.
#[cfg(feature = "reqwest")]
pub(crate) async fn get_file_from_tar_archive<R: tokio::io::AsyncRead + Unpin>(
    archive: &mut tokio_tar::Archive<R>,
    file_name: &Path,
) -> Result<Option<Vec<u8>>, ExtractError> {
    let target = crate::archive::normalize(file_name)?;
    let mut entries = archive.entries().map_err(ExtractError::IoError)?;
    while let Some(entry) = entries.next().await {
        let mut entry = entry.map_err(ExtractError::IoError)?;
        let path = entry.path().map_err(ExtractError::IoError)?;
        // Normalized comparison, matching the sparse path in `crate::archive`.
        if crate::archive::normalize(&path)? == target {
            drop(path);
            return crate::archive::read_raw_entry_contents(&mut entry)
                .await
                .map(Some);
        }
    }
    Ok(None)
}

#[cfg(test)]
mod test {
    use std::io::Write;

    use super::{extract_wheel, extract_wheel_via_buffering};

    /// Builds a minimal, but structurally realistic, wheel archive, with an
    /// executable bit set on the `.data/scripts/` entry (as a real console
    /// script binary shipped by a compiled wheel would have), to exercise
    /// the central-directory permission-fixup logic.
    fn build_test_wheel(path: &std::path::Path) {
        let file = std::fs::File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();

        zip.start_file("demo.py", options).unwrap();
        zip.write_all(b"print('hello')\n").unwrap();

        zip.start_file("demo-1.0.dist-info/RECORD", options)
            .unwrap();
        zip.write_all(b"demo.py,sha256=,16\ndemo-1.0.dist-info/RECORD,,\n")
            .unwrap();

        // A raw, pre-built executable shipped in `.data/scripts/`, as some
        // compiled wheels do (e.g. Rust/Go binaries), with the executable
        // bit set in the zip's central directory metadata.
        zip.start_file(
            "demo-1.0.data/scripts/demo-native-cli",
            options.unix_permissions(0o755),
        )
        .unwrap();
        zip.write_all(b"\0ELF-ish-binary-stand-in").unwrap();

        zip.start_file("demo-1.0.data/platlib/_demo_native.so", options)
            .unwrap();
        zip.write_all(b"not-really-a-shared-library").unwrap();

        zip.finish().unwrap();
    }

    async fn open(path: &std::path::Path) -> tokio::fs::File {
        tokio::fs::File::open(path).await.unwrap()
    }

    #[tokio::test]
    async fn test_extract_wheel_streaming_preserves_raw_layout_and_exec_bit() {
        let temp_dir = tempfile::tempdir().unwrap();
        let wheel_path = temp_dir.path().join("demo-1.0-py3-none-any.whl");
        build_test_wheel(&wheel_path);

        let destination = temp_dir.path().join("extracted");
        let result = extract_wheel(open(&wheel_path).await, &destination)
            .await
            .unwrap();
        assert!(result.total_size > 0);

        // Every entry is extracted at its original, archive-relative path -
        // no remapping is applied during extraction.
        assert!(destination.join("demo.py").is_file());
        assert!(destination.join("demo-1.0.dist-info/RECORD").is_file());
        let script_path = destination.join("demo-1.0.data/scripts/demo-native-cli");
        assert!(script_path.is_file());
        assert!(
            destination
                .join("demo-1.0.data/platlib/_demo_native.so")
                .is_file()
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&script_path)
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(
                mode & 0o111,
                0o111,
                "executable bit should have been recovered from the central directory"
            );
        }
    }

    #[tokio::test]
    async fn test_extract_wheel_via_buffering_preserves_raw_layout_and_exec_bit() {
        let temp_dir = tempfile::tempdir().unwrap();
        let wheel_path = temp_dir.path().join("demo-1.0-py3-none-any.whl");
        build_test_wheel(&wheel_path);

        let destination = temp_dir.path().join("extracted");
        let result = extract_wheel_via_buffering(open(&wheel_path).await, &destination)
            .await
            .unwrap();
        assert!(result.total_size > 0);

        assert!(destination.join("demo.py").is_file());
        let script_path = destination.join("demo-1.0.data/scripts/demo-native-cli");
        assert!(script_path.is_file());
        assert!(
            destination
                .join("demo-1.0.data/platlib/_demo_native.so")
                .is_file()
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&script_path)
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(
                mode & 0o111,
                0o111,
                "executable bit should have been preserved by the seek-based fallback"
            );
        }
    }
}
