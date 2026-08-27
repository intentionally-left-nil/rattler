//! Functionality to stream parts of a `.conda` archive for objects that implement both
//! [`std::io::Read`] and [`std::io::Seek`] like a [`std::fs::File`] or a [`std::io::Cursor<T>`].

use crate::ExtractError;
use crate::read::{stream_tar_bz2, stream_tar_zst};
use rattler_conda_types::package::CondaArchiveType;
use rattler_conda_types::package::PackageFile;
use std::fs::File;
use std::io::Write;
use std::{
    io::{Read, Seek, SeekFrom},
    path::Path,
};
use tar::Archive;
use zip::CompressionMethod;

fn stream_conda_zip_entry<'a, R: Read + Seek + 'a>(
    mut archive: zip::ZipArchive<R>,
    file_name: &str,
) -> Result<tar::Archive<impl Read + Sized + use<'a, R>>, ExtractError> {
    // Find the offset and size of the file in the zip.
    let (offset, size) = {
        let entry = archive.by_name(file_name)?;

        // Make sure the file is uncompressed.
        if entry.compression() != CompressionMethod::Stored {
            return Err(ExtractError::UnsupportedCompressionMethod);
        }

        (
            entry
                .data_start()
                .expect("data_start is available after reading entry"),
            entry.size(),
        )
    };

    // Seek to the position of the file
    let mut reader = archive.into_inner();
    reader.seek(SeekFrom::Start(offset))?;

    // Given the bytes in the zip archive of the file, decode it as a zst compressed tar file.
    stream_tar_zst(reader.take(size))
}

/// Stream the info section of a `.conda` package as a tar archive.
pub fn stream_conda_info<'a>(
    reader: impl Read + Seek + 'a,
) -> Result<tar::Archive<impl Read + Sized + 'a>, ExtractError> {
    let archive = zip::ZipArchive::new(reader)?;

    // Find the info entry in the archive
    let file_name = archive
        .file_names()
        .find(|file_name| file_name.starts_with("info-") && file_name.ends_with(".tar.zst"))
        .ok_or(ExtractError::MissingComponent)?
        .to_owned();

    stream_conda_zip_entry(archive, &file_name)
}

/// Stream the content section of a `.conda` package as a tar archive.
pub fn stream_conda_content<'a>(
    reader: impl Read + Seek + 'a,
) -> Result<tar::Archive<impl Read + Sized + 'a>, ExtractError> {
    let archive = zip::ZipArchive::new(reader)?;

    // Find the content entry in the archive
    let file_name = archive
        .file_names()
        .find(|file_name| file_name.starts_with("pkg-") && file_name.ends_with(".tar.zst"))
        .ok_or(ExtractError::MissingComponent)?
        .to_owned();

    stream_conda_zip_entry(archive, &file_name)
}

fn get_file_from_archive(
    archive: &mut Archive<impl Read>,
    file_name: &Path,
) -> Result<Vec<u8>, ExtractError> {
    for entry in archive.entries()? {
        let mut entry = entry?;
        if entry.path()? == file_name {
            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut buf)?;
            return Ok(buf);
        }
    }
    Err(ExtractError::MissingComponent)
}

/// Read a package file content from archive based on the path
pub fn read_package_file_content<'a>(
    file: impl Read + Seek + 'a,
    archive_type: CondaArchiveType,
    package_path: impl AsRef<Path>,
) -> Result<Vec<u8>, ExtractError> {
    match archive_type {
        CondaArchiveType::TarBz2 => {
            let mut archive = stream_tar_bz2(file);
            let buf = get_file_from_archive(&mut archive, package_path.as_ref())?;
            Ok(buf)
        }
        CondaArchiveType::Conda => {
            let mut info_archive = stream_conda_info(file)?;
            let buf = get_file_from_archive(&mut info_archive, package_path.as_ref())?;
            Ok(buf)
        }
    }
}

/// Read a package file from archive
/// Note: If you want to extract multiple `info/*` files then this will be slightly
///       slower than manually iterating over the archive entries with
///       custom logic as this skips over the rest of the archive
///
/// # Example
///
/// ```rust,no_run
/// use rattler_conda_types::package::AboutJson;
/// use rattler_package_streaming::seek::read_package_file;
///
/// let about_json = read_package_file::<AboutJson>("conda-forge/win-64/python-3.11.0-hcf16a7b_0_cpython.conda").unwrap();
/// ```
pub fn read_package_file<P: PackageFile>(path: impl AsRef<Path>) -> Result<P, ExtractError> {
    // stream extract the file from a package
    let file = File::open(&path)?;
    let content = read_package_file_content(
        &file,
        CondaArchiveType::try_from(&path).ok_or(ExtractError::UnsupportedArchiveType)?,
        P::package_path(),
    )?;

    P::from_slice(&content)
        .map_err(|e| ExtractError::ArchiveMemberParseError(P::package_path().to_owned(), e))
}

/// Extracts the contents of a wheel archive (a plain zip file) into
/// `destination`, preserving each entry's original archive-relative path
/// (i.e. the same layout `unzip` would produce). See the module-level
/// documentation of [`rattler_conda_types::package::wheel`] for why the
/// `site-packages`/`python-scripts` remapping is deliberately *not* applied
/// here.
///
/// Unlike `.conda`/`.tar.bz2` extraction this does not stream: it requires a
/// [`Seek`]-able reader since the underlying [`zip::ZipArchive`] needs random
/// access to read the central directory. Callers that only have a streaming
/// (e.g. HTTP) source should buffer the archive first (see
/// [`crate::tokio::async_read::extract_wheel_via_buffering`]).
pub fn extract_wheel_contents<R: Read + Seek>(
    reader: R,
    destination: &Path,
) -> Result<(), ExtractError> {
    std::fs::create_dir_all(destination).map_err(ExtractError::CouldNotCreateDestination)?;

    let mut archive = zip::ZipArchive::new(reader)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        if entry.is_dir() {
            continue;
        }
        let Some(enclosed) = entry.enclosed_name() else {
            // Skip entries with unsafe (e.g. zip-slip) paths.
            continue;
        };
        let out_path = destination.join(&enclosed);
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut out_file = File::create(&out_path)?;
        std::io::copy(&mut entry, &mut out_file)?;

        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            // Best-effort: a failure to set permissions should not fail the
            // whole extraction.
            let _ = std::fs::set_permissions(&out_path, std::fs::Permissions::from_mode(mode));
        }
    }

    Ok(())
}

/// Get a [`PackageFile`] from temporary archive and extract it to a writer
pub fn extract_package_file<'a, P: PackageFile>(
    reader: impl Read + Seek + 'a,
    location: &Path,
    writer: &mut impl Write,
) -> Result<(), ExtractError> {
    let content = read_package_file_content(
        reader,
        CondaArchiveType::try_from(location).ok_or(ExtractError::UnsupportedArchiveType)?,
        P::package_path(),
    )?;

    writer.write_all(&content)?;

    writer.flush()?;

    Ok(())
}
