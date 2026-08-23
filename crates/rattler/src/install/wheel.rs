//! Support for installing Python wheels (`.whl`) that are referenced through
//! a `v3.whl` ("wheel") entry in conda repodata, so that they end up
//! installed and tracked in `conda-meta` exactly like any other conda
//! package - `conda list`/`rattler list`, update, and uninstall all work
//! without any special-casing.
//!
//! ## How this works
//!
//! Installing a wheel is structurally quite different from installing a
//! `.conda`/`.tar.bz2` package: there is no `info/index.json` or
//! `info/paths.json` inside the archive, and the files inside a wheel are
//! laid out according to [the wheel
//! spec](https://packaging.python.org/en/latest/specifications/binary-distribution-format/)
//! (`purelib`/`platlib`/`scripts`/`data`/`headers` categories) rather than
//! being pre-arranged for a specific target environment.
//!
//! Rather than duplicating rattler's existing (and well tested) `noarch:
//! python` install machinery - path remapping into `site-packages`,
//! `.dist-info` handling, entry point generation, prefix registration,
//! clobber detection, etc. - this module *synthesizes* the equivalent
//! in-memory `IndexJson`/`PathsJson`/`LinkJson` from the wheel's own `RECORD`
//! and `entry_points.txt` files (see
//! [`rattler_conda_types::package::wheel`]) and feeds them into
//! [`crate::install::link_package_sync`] through
//! [`crate::install::InstallOptions`] - exactly like the [`super::Installer`]
//! already does for conda packages read from the on-disk package cache.
//!
//! As a result, a wheel's contents are always linked using the
//! noarch-python code path (ending up in the environment's real
//! `site-packages`/`bin` directories), *regardless* of whether the wheel
//! itself is platform-independent (a `noarch: python`, `py3-none-any` wheel)
//! or platform/ABI-specific (a compiled wheel with a real `subdir`, e.g.
//! `linux-64`, and a platform tag such as `manylinux_x86_64`). The wheel's
//! own `PackageRecord.noarch`/`subdir` - as it comes from repodata - is left
//! untouched, and continues to drive solving and
//! [`super::Transaction`] diffing exactly like for any other conda
//! package: only the *installation strategy* differs, entirely as an
//! implementation detail of this module.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use rattler_conda_types::{
    NoArchType, RepoDataRecord,
    package::{
        EntryPoint, IndexJson, LinkJson, NoArchLinks, PathType, PathsEntry, PathsJson,
        PythonEntryPoints,
        wheel::{WheelRecord, find_dist_info_dir, parse_console_scripts},
    },
    prefix::Prefix,
    prefix_record::{self, Link, LinkType},
    utils::{InvalidPathComponentError, ensure_safe_path_component},
};
use rattler_networking::LazyClient;
use rattler_package_streaming::ExtractError;

use super::{InstallDriver, InstallError, InstallOptions, clobber_registry::ClobberRegistry};

/// Errors that might occur while installing a wheel.
#[derive(Debug, thiserror::Error)]
pub enum WheelInstallError {
    /// The wheel could not be downloaded or extracted.
    #[error("failed to fetch wheel from '{0}'")]
    FailedToFetch(String, #[source] Box<ExtractError>),

    /// The wheel's `RECORD` file could not be read or parsed.
    #[error("failed to read the wheel's 'RECORD' file")]
    FailedToReadRecord(#[source] std::io::Error),

    /// The wheel's `entry_points.txt` file could not be read.
    #[error("failed to read the wheel's 'entry_points.txt' file")]
    FailedToReadEntryPoints(#[source] std::io::Error),

    /// A record contained metadata that could be used to escape the cache or
    /// installation prefix.
    #[error(transparent)]
    UnsafePackageRecord(#[from] InvalidPathComponentError),

    /// Linking the wheel's files into the prefix failed.
    #[error(transparent)]
    Install(#[from] InstallError),

    /// A generic IO error occurred.
    #[error("an io error occurred")]
    Io(#[from] std::io::Error),
}

/// Computes the name of the directory, relative to the wheel cache root, in
/// which an extracted wheel is (or should be) stored.
///
/// The `_whl` suffix keeps the directory namespace distinct from the
/// `name-version-build` directories used by the regular conda package cache,
/// even though wheels are cached separately.
fn wheel_cache_dir_name(record: &RepoDataRecord) -> Result<String, InvalidPathComponentError> {
    let name = record.package_record.name.as_normalized();
    let version = record.package_record.version.to_string();
    let build = &record.package_record.build;
    ensure_safe_path_component(name)?;
    ensure_safe_path_component(&version)?;
    ensure_safe_path_component(build)?;
    Ok(format!("{name}-{version}-{build}_whl"))
}

/// Downloads (if necessary) and extracts the wheel referenced by `record`
/// into a directory under `cache_dir`, returning the directory that contains
/// the extracted wheel contents.
///
/// If a previous extraction is already present (recognized by the presence
/// of a `.dist-info` directory) it is reused as-is, mirroring the
/// [`rattler_cache::validation::ValidationMode::Skip`] default used for the
/// regular conda package cache: this is a presence check, not a full
/// hash-verification of every file.
pub async fn populate_wheel_cache(
    record: &RepoDataRecord,
    cache_dir: &Path,
    downloader: LazyClient,
) -> Result<PathBuf, WheelInstallError> {
    let dir_name = wheel_cache_dir_name(record)?;
    let destination = cache_dir.join(&dir_name);

    if find_dist_info_dir(&destination).is_ok() {
        return Ok(destination);
    }

    fs_err::tokio::create_dir_all(cache_dir).await?;

    let temp_dir = tempfile::Builder::new()
        .prefix(&format!(".{dir_name}"))
        .tempdir_in(cache_dir)?;

    if record.url.scheme() == "file" {
        let path = record.url.to_file_path().map_err(|()| {
            WheelInstallError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("could not convert '{}' to a file path", record.url),
            ))
        })?;
        rattler_package_streaming::tokio::fs::extract_wheel(&path, temp_dir.path())
            .await
            .map_err(|e| WheelInstallError::FailedToFetch(record.url.to_string(), Box::new(e)))?;
    } else {
        rattler_package_streaming::reqwest::tokio::extract_wheel(
            downloader.client().clone(),
            record.url.clone(),
            temp_dir.path(),
            record.package_record.sha256,
            None,
        )
        .await
        .map_err(|e| WheelInstallError::FailedToFetch(record.url.to_string(), Box::new(e)))?;
    }

    // Take ownership of the temp directory so it isn't cleaned up, then
    // atomically move it into place, exactly like the regular package cache
    // does for conda packages.
    let temp_path = temp_dir.keep();
    if destination.is_dir() {
        fs_err::tokio::remove_dir_all(&destination).await?;
    }
    fs_err::tokio::rename(&temp_path, &destination).await?;

    Ok(destination)
}

/// Reads the console-script entry points declared in the wheel's
/// `entry_points.txt` file (if any). Lines that cannot be parsed as a valid
/// [`EntryPoint`] are skipped with a warning rather than failing the whole
/// installation, since entry point generation is best-effort metadata on top
/// of the actual installed files.
fn read_console_script_entry_points(
    cached_wheel_dir: &Path,
) -> Result<Vec<EntryPoint>, WheelInstallError> {
    let dist_info_dir =
        find_dist_info_dir(cached_wheel_dir).map_err(WheelInstallError::FailedToReadRecord)?;
    let entry_points_path = dist_info_dir.join("entry_points.txt");

    let content = match fs_err::read_to_string(&entry_points_path) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(WheelInstallError::FailedToReadEntryPoints(e)),
    };

    let mut entry_points = Vec::new();
    for raw in parse_console_scripts(&content) {
        match raw.parse::<EntryPoint>() {
            Ok(entry_point) => entry_points.push(entry_point),
            Err(e) => {
                tracing::warn!(
                    "skipping invalid console_scripts entry point '{raw}' in {}: {e}",
                    entry_points_path.display()
                );
            }
        }
    }
    Ok(entry_points)
}

/// Builds the synthetic `IndexJson` that is fed into
/// [`crate::install::link_package_sync`] for wheel installs.
///
/// Only the `name` and `noarch` fields are functionally relevant here:
/// `name` must match the wheel's real package name because
/// [`crate::install::clobber_registry::ClobberRegistry::register_paths`]
/// records installed paths under `index_json.name`, and that registry is
/// later reconciled against the real package names read back from
/// `conda-meta` - so anything else here would be treated as a "phantom"
/// package. `noarch` drives the `site-packages`/`python-scripts` path
/// remapping and entry point generation. None of the other fields are read
/// by the linker for a package whose `paths.json`/`index.json`/`link.json`
/// are all provided in-memory (rather than read from disk), and none of this
/// synthetic data is ever persisted: the real `PrefixRecord` written to
/// `conda-meta` is built from the actual repodata [`RepoDataRecord`], not
/// from this value.
fn synthetic_index_json(record: &rattler_conda_types::PackageRecord) -> IndexJson {
    IndexJson {
        arch: None,
        build: record.build.clone(),
        build_number: record.build_number,
        constrains: Vec::new(),
        depends: Vec::new(),
        extra_depends: std::collections::BTreeMap::default(),
        features: None,
        flags: Vec::new(),
        license: None,
        license_family: None,
        name: record.name.clone(),
        // Always use the noarch-python linking convention so wheel content
        // is placed into the environment's real `site-packages`/`bin`
        // directories, regardless of whether the wheel itself is
        // platform-independent or a compiled, subdir-specific wheel.
        noarch: NoArchType::python(),
        platform: None,
        purls: None,
        python_site_packages_path: None,
        repodata_revision: None,
        subdir: None,
        timestamp: None,
        track_features: Vec::new(),
        version: record.version.clone(),
    }
}

/// Links an already-cached (extracted) wheel from `cached_wheel_dir` into
/// `target_dir`, returning the [`PathsEntry`] list describing every
/// installed file (for use in a `PrefixRecord`) and the [`LinkType`] that
/// was used, mirroring [`crate::install::link_package_sync`].
///
/// `package_record` is the wheel's *real* metadata as it comes from
/// repodata; its `name` is used to register the installed paths with the
/// clobber registry (see [`synthetic_index_json`]) so that later
/// reconciliation against `conda-meta` finds the same package name.
pub fn link_wheel_sync(
    cached_wheel_dir: &Path,
    package_record: &rattler_conda_types::PackageRecord,
    target_dir: &Prefix,
    clobber_registry: Arc<Mutex<ClobberRegistry>>,
    mut options: InstallOptions,
) -> Result<(Vec<prefix_record::PathsEntry>, LinkType), WheelInstallError> {
    let wheel_record = WheelRecord::from_extracted_directory(cached_wheel_dir)
        .map_err(WheelInstallError::FailedToReadRecord)?;

    let paths = wheel_record
        .entries
        .iter()
        .map(|entry| PathsEntry {
            relative_path: entry.mapped_path(),
            no_link: false,
            path_type: PathType::HardLink,
            prefix_placeholder: None,
            sha256: entry.sha256,
            size_in_bytes: entry.size,
        })
        .collect();

    let entry_points = read_console_script_entry_points(cached_wheel_dir)?;
    let link_json = if entry_points.is_empty() {
        None
    } else {
        Some(LinkJson {
            noarch: NoArchLinks::Python(PythonEntryPoints { entry_points }),
            package_metadata_version: 1,
        })
    };

    options.paths_json = Some(PathsJson {
        paths,
        paths_version: 1,
    });
    options.index_json = Some(synthetic_index_json(package_record));
    options.link_json = Some(link_json);

    super::link_package_sync(cached_wheel_dir, target_dir, clobber_registry, options)
        .map_err(WheelInstallError::Install)
}

/// Downloads/extracts (if necessary) and installs the wheel described by
/// `record` into `target_prefix`, writing a `PrefixRecord` to `conda-meta` in
/// exactly the same way as for a regular conda package. This is the wheel
/// equivalent of the internal `link_package` helper used by
/// [`super::Installer`] for conda packages.
#[allow(clippy::too_many_arguments)]
pub async fn install_wheel(
    record: &RepoDataRecord,
    target_prefix: &Prefix,
    wheel_cache_dir: &Path,
    downloader: LazyClient,
    install_options: InstallOptions,
    driver: &InstallDriver,
    requested_specs: Vec<String>,
) -> Result<(), WheelInstallError> {
    let cached_wheel_dir = populate_wheel_cache(record, wheel_cache_dir, downloader).await?;

    let record = record.clone();
    let target_prefix = target_prefix.clone();
    let clobber_registry = driver.clobber_registry.clone();
    let conda_meta_path = target_prefix.path().join("conda-meta");
    let cached_wheel_dir_for_task = cached_wheel_dir.clone();

    let (tx, rx) = tokio::sync::oneshot::channel();
    rayon::spawn_fifo(move || {
        let inner = move || -> Result<(), WheelInstallError> {
            let (paths, link_type) = link_wheel_sync(
                &cached_wheel_dir_for_task,
                &record.package_record,
                &target_prefix,
                clobber_registry,
                install_options,
            )?;

            let prefix_record = prefix_record::PrefixRecord {
                extracted_package_dir: Some(cached_wheel_dir_for_task.clone()),
                link: Some(Link {
                    source: cached_wheel_dir_for_task,
                    link_type: Some(link_type),
                }),
                requested_specs,
                ..prefix_record::PrefixRecord::from_repodata_record(record.clone(), paths)
            };

            let pkg_meta_path = prefix_record.file_name();
            prefix_record
                .write_to_path(conda_meta_path.join(&pkg_meta_path), true)
                .map_err(WheelInstallError::Io)?;

            Ok(())
        };
        let _ = tx.send(inner());
    });

    rx.await
        .map_err(|_recv_err| WheelInstallError::Install(InstallError::Cancelled))?
}

#[cfg(test)]
mod test {
    use std::{io::Write, str::FromStr};

    use rattler_conda_types::{
        NoArchType, PackageName, PackageRecord, Platform, PrefixRecord, RepoDataRecord, Version,
        package::{ArchiveIdentifier, DistArchiveIdentifier, WheelArchiveType},
        prefix::Prefix,
    };
    use url::Url;

    use crate::install::{Installer, PythonInfo};

    /// Builds a small, but structurally realistic, wheel containing:
    /// - a root-level module (`demo.py`) -> lands in `site-packages`
    /// - a `console_scripts` entry point (`demo-cli`) -> generates a launcher
    ///   script in the environment's `bin`/`Scripts` directory
    /// - a `.data/platlib/` file, simulating the compiled payload of a
    ///   subdir-specific wheel -> also lands in `site-packages`
    /// - a `.data/scripts/` file -> lands in the environment's `bin`
    ///   directory alongside the generated entry point
    fn build_test_wheel(path: &std::path::Path) {
        let file = std::fs::File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default();

        zip.start_file("demo.py", options).unwrap();
        zip.write_all(b"def main():\n    print('hello')\n").unwrap();

        zip.start_file("demo-1.0.dist-info/METADATA", options)
            .unwrap();
        zip.write_all(b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n")
            .unwrap();

        zip.start_file("demo-1.0.dist-info/RECORD", options)
            .unwrap();
        zip.write_all(
            b"demo.py,sha256=,29\n\
              demo-1.0.data/scripts/demo-data-script,sha256=,0\n\
              demo-1.0.data/platlib/_demo_native.so,sha256=,0\n\
              demo-1.0.dist-info/METADATA,,\n\
              demo-1.0.dist-info/RECORD,,\n",
        )
        .unwrap();

        zip.start_file("demo-1.0.dist-info/entry_points.txt", options)
            .unwrap();
        zip.write_all(b"[console_scripts]\ndemo-cli = demo:main\n")
            .unwrap();

        zip.start_file("demo-1.0.data/scripts/demo-data-script", options)
            .unwrap();
        zip.write_all(b"").unwrap();

        zip.start_file("demo-1.0.data/platlib/_demo_native.so", options)
            .unwrap();
        zip.write_all(b"").unwrap();

        zip.finish().unwrap();
    }

    /// Constructs a `RepoDataRecord` for the fake `python` "package" used to
    /// give the installer a `PythonInfo` without needing to actually
    /// download/link a real Python interpreter.
    fn fake_python_repodata_record() -> RepoDataRecord {
        let package_record = PackageRecord::new(
            PackageName::new_unchecked("python"),
            Version::from_str("3.11.0").unwrap(),
            "h1234_0".to_string(),
        );
        RepoDataRecord {
            package_record,
            identifier: "python-3.11.0-h1234_0.conda".parse().unwrap(),
            url: Url::parse("https://example.com/python-3.11.0-h1234_0.conda").unwrap(),
            channel: None,
        }
    }

    /// Constructs a `RepoDataRecord` that points at a local wheel file,
    /// exactly as it would be produced from a `v3.whl` repodata entry (see
    /// [`rattler_conda_types::WhlPackageRecord`]) after being resolved to an
    /// absolute URL. `noarch` is intentionally left unset and `subdir` is set
    /// to a real platform to exercise the "subdir-specific (compiled) wheel"
    /// case end-to-end.
    fn wheel_repodata_record(wheel_path: &std::path::Path) -> RepoDataRecord {
        let mut package_record = PackageRecord::new(
            PackageName::new_unchecked("demo"),
            Version::from_str("1.0").unwrap(),
            "py3_none_any_0".to_string(),
        );
        package_record.noarch = NoArchType::none();
        package_record.subdir = "linux-64".to_string();

        RepoDataRecord {
            package_record,
            identifier: DistArchiveIdentifier::new(
                ArchiveIdentifier {
                    name: "demo".to_string(),
                    version: "1.0".to_string(),
                    build_string: "py3_none_any_0".to_string(),
                },
                WheelArchiveType::Whl,
            ),
            url: Url::from_file_path(wheel_path).unwrap(),
            channel: None,
        }
    }

    #[tokio::test]
    async fn test_install_and_uninstall_wheel_end_to_end() {
        let temp_dir = tempfile::tempdir().unwrap();
        let wheel_path = temp_dir.path().join("demo-1.0-py3-none-any.whl");
        build_test_wheel(&wheel_path);

        let prefix_dir = temp_dir.path().join("prefix");
        let prefix = Prefix::create(&prefix_dir).unwrap();

        let python_record = fake_python_repodata_record();
        let python_prefix_record =
            PrefixRecord::from_repodata_record(python_record.clone(), Vec::new());
        let wheel_record = wheel_repodata_record(&wheel_path);

        // Pretend that `python` is already installed by writing its
        // `conda-meta` entry directly (rather than going through the
        // installer, since installing a real `python` package is out of
        // scope for this test - only its presence in `conda-meta` and
        // metadata matter for deriving a `PythonInfo`).
        let conda_meta_dir = prefix_dir.join("conda-meta");
        std::fs::create_dir_all(&conda_meta_dir).unwrap();
        python_prefix_record
            .write_to_path(conda_meta_dir.join(python_prefix_record.file_name()), true)
            .unwrap();

        // --- Install ---
        let result = Installer::new()
            .with_wheel_cache_dir(temp_dir.path().join("wheel-cache"))
            .install(&prefix, vec![python_record.clone(), wheel_record.clone()])
            .await
            .expect("installing a wheel should succeed");
        assert_eq!(result.transaction.packages_to_install(), 1);

        // A `conda-meta` entry should exist for the wheel, with full paths
        // tracked, just like a regular conda package.
        let conda_meta_path = prefix_dir
            .join("conda-meta")
            .join("demo-1.0-py3_none_any_0.json");
        assert!(conda_meta_path.is_file(), "conda-meta entry should exist");

        let installed_record = PrefixRecord::from_path(&conda_meta_path).unwrap();
        assert_eq!(
            installed_record
                .repodata_record
                .package_record
                .name
                .as_normalized(),
            "demo"
        );
        // The wheel's own (real, repodata-derived) subdir/noarch metadata is
        // preserved verbatim - only the *linking strategy* used the
        // noarch-python convention internally.
        assert_eq!(
            installed_record.repodata_record.package_record.subdir,
            "linux-64"
        );
        assert!(
            installed_record
                .repodata_record
                .package_record
                .noarch
                .is_none()
        );
        // Every wheel-owned file should be tracked for uninstall/listing: the
        // 5 files listed in `RECORD` plus the generated `demo-cli` entry
        // point script (1 file on unix, 2 on windows).
        let expected_paths = if cfg!(windows) { 7 } else { 6 };
        assert_eq!(installed_record.paths_data.paths.len(), expected_paths);

        let python_info = PythonInfo::from_version(
            &Version::from_str("3.11.0").unwrap(),
            None,
            Platform::current(),
        )
        .unwrap();

        // The root-level module landed in the real site-packages directory.
        assert!(
            prefix_dir
                .join(&python_info.site_packages_path)
                .join("demo.py")
                .is_file()
        );
        // The `.data/platlib` file (simulating compiled content) also landed
        // in site-packages.
        assert!(
            prefix_dir
                .join(&python_info.site_packages_path)
                .join("_demo_native.so")
                .is_file()
        );
        // The `.data/scripts` file landed in the environment's bin directory.
        assert!(
            prefix_dir
                .join(&python_info.bin_dir)
                .join("demo-data-script")
                .is_file()
        );
        // The console_scripts entry point was generated in the bin directory
        // too.
        let entry_point_path = prefix_dir.join(&python_info.bin_dir).join("demo-cli");
        assert!(
            entry_point_path.is_file() || entry_point_path.with_extension("exe").is_file(),
            "entry point script should have been generated"
        );

        // --- Uninstall ---
        let result = Installer::new()
            .with_wheel_cache_dir(temp_dir.path().join("wheel-cache"))
            .install(&prefix, vec![python_record])
            .await
            .expect("uninstalling a wheel should succeed");
        assert_eq!(result.transaction.packages_to_uninstall(), 1);

        assert!(
            !conda_meta_path.exists(),
            "conda-meta entry should have been removed"
        );
        assert!(
            !prefix_dir
                .join(&python_info.site_packages_path)
                .join("demo.py")
                .exists()
        );
        assert!(
            !prefix_dir
                .join(&python_info.site_packages_path)
                .join("_demo_native.so")
                .exists()
        );
        assert!(
            !prefix_dir
                .join(&python_info.bin_dir)
                .join("demo-data-script")
                .exists()
        );
    }
}
