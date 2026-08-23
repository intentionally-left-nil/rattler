//! Functions that enable extracting or streaming a Conda package for objects
//! that implement the [`tokio::io::AsyncRead`] + [`tokio::io::AsyncSeek`] traits.

use std::path::Path;

use async_zip::base::read::seek::ZipFileReader;
use tokio::io::{AsyncRead, AsyncSeek};
use tokio_util::compat::{FuturesAsyncReadCompatExt, TokioAsyncReadCompatExt};

use crate::ExtractError;

use super::shared::{DEFAULT_BUF_SIZE, extract_tar_zst_entry};

/// Extracts the contents of a `.conda` package archive using the seek-based API.
/// This is more efficient than streaming when the entire file is available (e.g., from disk or memory).
///
/// This function only performs extraction and does NOT compute hashes or track size.
/// Use this when you've already computed hashes separately or don't need them.
pub async fn extract_conda(
    reader: impl AsyncRead + AsyncSeek + Send + Unpin + 'static,
    destination: &Path,
) -> Result<(), ExtractError> {
    // Ensure the destination directory exists
    tokio::fs::create_dir_all(destination)
        .await
        .map_err(ExtractError::CouldNotCreateDestination)?;

    // Clone destination for the async block
    let destination = destination.to_owned();

    // Convert to futures traits for async_zip (which uses futures traits)
    let mut compat_reader = reader.compat();
    let mut buf_reader =
        futures::io::BufReader::with_capacity(DEFAULT_BUF_SIZE, &mut compat_reader);

    // Create a ZIP reader using the seek API
    let mut zip_reader = ZipFileReader::new(&mut buf_reader)
        .await
        .map_err(|e| ExtractError::IoError(std::io::Error::other(e)))?;

    // Process each ZIP entry
    let num_entries = zip_reader.file().entries().len();
    for index in 0..num_entries {
        let entry = zip_reader.file().entries().get(index).ok_or_else(|| {
            ExtractError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "entry not found",
            ))
        })?;

        let filename = entry.filename().as_str().map_err(|e| {
            ExtractError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;

        // Only extract .tar.zst files
        if filename.ends_with(".tar.zst") {
            let entry_reader = zip_reader
                .reader_with_entry(index)
                .await
                .map_err(|e| ExtractError::IoError(std::io::Error::other(e)))?;

            // Convert from futures traits to tokio traits
            let mut compat_entry = entry_reader.compat();
            extract_tar_zst_entry(&mut compat_entry, &destination).await?;
        }
    }

    Ok(())
}

/// Extracts the contents of a wheel (`.whl`) archive using the seek-based
/// API. This is more efficient than streaming when the entire file is
/// already available (e.g. buffered via [`crate::tokio::async_read::extract_wheel_via_buffering`]),
/// since the central directory (which is required to know each entry's
/// Unix executable permission bits) is available upfront rather than only
/// after streaming through every entry.
///
/// This function only performs extraction and does NOT compute hashes or
/// track size. Use this when you've already computed hashes separately.
pub async fn extract_wheel(
    reader: impl AsyncRead + AsyncSeek + Send + Unpin + 'static,
    destination: &Path,
) -> Result<(), ExtractError> {
    // Ensure the destination directory exists
    tokio::fs::create_dir_all(destination)
        .await
        .map_err(ExtractError::CouldNotCreateDestination)?;

    // Clone destination for the async block
    let destination = destination.to_owned();

    // Convert to futures traits for async_zip (which uses futures traits)
    let mut compat_reader = reader.compat();
    let mut buf_reader =
        futures::io::BufReader::with_capacity(DEFAULT_BUF_SIZE, &mut compat_reader);

    // Create a ZIP reader using the seek API. Unlike the streaming reader,
    // this immediately parses the central directory, so every entry's
    // metadata (including Unix permission bits) is available upfront.
    let mut zip_reader = ZipFileReader::new(&mut buf_reader)
        .await
        .map_err(|e| ExtractError::IoError(std::io::Error::other(e)))?;

    let num_entries = zip_reader.file().entries().len();
    for index in 0..num_entries {
        let (relpath, is_dir, mode) = {
            let entry = zip_reader.file().entries().get(index).ok_or_else(|| {
                ExtractError::IoError(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "entry not found",
                ))
            })?;

            let filename = entry.filename().as_str().map_err(|e| {
                ExtractError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            })?;

            let Some(relpath) = super::shared::sanitize_zip_entry_name(filename) else {
                tracing::warn!("skipping unsafe wheel entry path: {filename}");
                continue;
            };

            (
                relpath,
                entry.dir().unwrap_or(false),
                entry.unix_permissions(),
            )
        };

        let out_path = destination.join(
            rattler_conda_types::package::wheel::map_wheel_archive_path(&relpath),
        );

        if is_dir {
            tokio::fs::create_dir_all(&out_path)
                .await
                .map_err(ExtractError::IoError)?;
            continue;
        }

        if let Some(parent) = out_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(ExtractError::IoError)?;
        }

        let entry_reader = zip_reader
            .reader_with_entry(index)
            .await
            .map_err(|e| ExtractError::IoError(std::io::Error::other(e)))?;
        let mut compat_entry = entry_reader.compat();
        let mut out_file = tokio::fs::File::create(&out_path)
            .await
            .map_err(ExtractError::IoError)?;
        tokio::io::copy(&mut compat_entry, &mut out_file)
            .await
            .map_err(ExtractError::IoError)?;

        super::shared::apply_executable_bit(mode.map(u32::from), &out_path).await?;
    }

    Ok(())
}
