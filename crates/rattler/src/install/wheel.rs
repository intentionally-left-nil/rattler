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
//!
//! ## Caching
//!
//! Wheels are fetched and cached through the *same*
//! [`rattler_cache::package_cache::PackageCache`] machinery conda packages
//! use - cross-process locking, in-process de-duplication of concurrent
//! identical fetches, a shared download-concurrency semaphore, retry on
//! transient failure, whole-archive hash verification with
//! delete-and-retry on mismatch, and atomic (tempdir-then-rename)
//! extraction - just parameterized with a wheel-flavored extractor (see
//! [`populate_wheel_cache`]) and directory validator (see
//! [`wheel_validator`]) instead of the conda-specific ones. Wheels are
//! kept in their own [`rattler_cache::package_cache::PackageCache`]
//! instance, rooted at a separate directory (see
//! [`super::Installer::with_wheel_cache_dir`]), so a wheel can never
//! collide with a conda package cache entry that happens to share the same
//! synthesized name/version/build string.
//!
//! The cache holds a faithful, archive-relative unpack of the wheel (the
//! same layout `unzip` would produce) - *not* remapped onto the
//! `site-packages`/`python-scripts` install convention. That remapping is
//! applied only when computing each file's install-time destination (see
//! [`crate::install::InstallOptions::is_wheel`]), which keeps the on-disk
//! cache format independent of the remapping convention and lets the
//! wheel's own `RECORD` hashes be validated directly against the cached
//! files.
//!
//! A caller that wants to check whether a wheel is already cached (e.g. to
//! build an offline-install exclusion set the way
//! `rattler_solve`'s test suite does for conda packages via
//! [`rattler_cache::package_cache::PackageCache::index`]) must query a
//! `PackageCache` instance built the same way [`super::Installer::install`]
//! builds its wheel cache - i.e. rooted at the wheel cache directory with a
//! [`wheel_validator`]-validated layer - not the conda package cache
//! instance: [`rattler_cache::package_cache::CacheIndex::contains_record`]
//! answers correctly for whichever archive kind the queried `PackageCache`
//! instance was actually built for, and a wheel is never present in a
//! conda-flavored instance's layers (nor vice versa).

use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use rattler_cache::package_cache::{CacheReporter, DirValidator, PackageCache, PackageCacheError};
use rattler_conda_types::{
    NoArchType, Platform, RepoDataRecord,
    package::{
        EntryPoint, IndexJson, LinkJson, NoArchLinks, PathType, PathsEntry, PathsJson,
        PythonEntryPoints,
        wheel::{
            WheelRecord, find_dist_info_dir, is_wheel_script_path, map_wheel_archive_path,
            parse_console_scripts,
        },
    },
    prefix::Prefix,
    prefix_record::{self, Link, LinkType},
};
use rattler_networking::{LazyClient, retry_policies::default_retry_policy};

use super::{
    InstallDriver, InstallError, InstallOptions, PythonInfo, clobber_registry::ClobberRegistry,
};

/// Errors that might occur while installing a wheel.
#[derive(Debug, thiserror::Error)]
pub enum WheelInstallError {
    /// The wheel could not be downloaded, cached, or extracted.
    #[error("failed to fetch wheel from '{0}'")]
    FailedToFetch(String, #[source] Box<PackageCacheError>),

    /// The wheel's `RECORD` file could not be read or parsed.
    #[error("failed to read the wheel's 'RECORD' file")]
    FailedToReadRecord(#[source] std::io::Error),

    /// The wheel's `entry_points.txt` file could not be read.
    #[error("failed to read the wheel's 'entry_points.txt' file")]
    FailedToReadEntryPoints(#[source] std::io::Error),

    /// Linking the wheel's files into the prefix failed.
    #[error(transparent)]
    Install(#[from] InstallError),

    /// A generic IO error occurred.
    #[error("an io error occurred")]
    Io(#[from] std::io::Error),
}

/// The [`DirValidator`] used for the wheel [`rattler_cache::package_cache::PackageCacheLayer`],
/// validating an extracted wheel directory against its own `RECORD`
/// manifest (see [`rattler_cache::validation::validate_wheel_directory`])
/// instead of a conda package's `info/index.json`/`info/paths.json`.
///
/// A successful validation never carries `IndexJson`/`PathsJson` data back
/// (unlike the conda validator): a wheel's synthesized equivalent is cheap
/// to reconstruct from its `RECORD` file at install time (see
/// [`link_wheel_sync`]), so there is nothing worth caching here.
pub fn wheel_validator() -> DirValidator {
    Arc::new(|path, mode| {
        rattler_cache::validation::validate_wheel_directory(path, mode)
            .map(|()| (None, None))
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    })
}

/// Downloads (if necessary) and extracts the wheel referenced by `record`,
/// returning the directory that contains the extracted wheel contents.
///
/// This goes through the *same* [`PackageCache`] machinery a conda package
/// fetch does - see the module-level "Caching" section - parameterized with
/// [`rattler_package_streaming`]'s wheel extractor instead of its
/// `.conda`/`.tar.bz2` one. `wheel_cache` must have been constructed with a
/// [`wheel_validator`]-validated layer (see
/// [`super::Installer::with_wheel_cache_dir`]).
pub async fn populate_wheel_cache(
    record: &RepoDataRecord,
    wheel_cache: &PackageCache,
    downloader: LazyClient,
    reporter: Option<Arc<dyn CacheReporter>>,
    concurrent_requests_semaphore: Option<Arc<tokio::sync::Semaphore>>,
) -> Result<PathBuf, WheelInstallError> {
    let cache_metadata = if record.url.scheme() == "file" {
        let path = record.url.to_file_path().map_err(|()| {
            WheelInstallError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("could not convert '{}' to a file path", record.url),
            ))
        })?;

        wheel_cache
            .get_or_fetch_from_path_with_extractor(
                &record.package_record,
                &path,
                reporter,
                |archive, destination| async move {
                    rattler_package_streaming::tokio::fs::extract_wheel(&archive, &destination)
                        .await
                },
            )
            .await
    } else {
        wheel_cache
            .get_or_fetch_from_url_with_extractor(
                &record.package_record,
                record.url.clone(),
                downloader,
                default_retry_policy(),
                reporter,
                concurrent_requests_semaphore,
                |client, url, destination: PathBuf, sha256, reporter| async move {
                    rattler_package_streaming::reqwest::tokio::extract_wheel(
                        client,
                        url,
                        &destination,
                        sha256,
                        reporter,
                    )
                    .await
                },
            )
            .await
    }
    .map_err(|e| WheelInstallError::FailedToFetch(record.url.to_string(), Box::new(e)))?;

    Ok(cache_metadata.path().to_path_buf())
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

    // `relative_path` here is the *archive*-relative path (the cache holds a
    // faithful, archive-relative unpack - see the module-level "Caching"
    // section), exactly like a conda package's own `info/paths.json` uses a
    // path relative to the package directory. `options.is_wheel` (set below)
    // tells `compute_paths` to apply the `site-packages`/`python-scripts`
    // remapping on top of this before resolving the install-time
    // destination, instead of assuming it's already been applied the way a
    // conda-build-produced `paths.json` would be.
    let paths = wheel_record
        .entries
        .iter()
        .map(|entry| PathsEntry {
            relative_path: entry.archive_path.clone(),
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
    options.is_wheel = true;

    // Extract what the shebang-fixup pass below needs before `options` (and
    // its `target_prefix`/`python_info`) is consumed by `link_package_sync`.
    let python_info = options.python_info.clone();
    let target_prefix = options
        .target_prefix
        .clone()
        .unwrap_or_else(|| target_dir.path().to_path_buf());
    let platform = options.platform.unwrap_or(Platform::current());

    let (paths, link_type) =
        super::link_package_sync(cached_wheel_dir, target_dir, clobber_registry, options)
            .map_err(WheelInstallError::Install)?;

    // Rewrite any `#!python`/`#!pythonw` placeholder shebang (see the wheel
    // spec's "scripts" category) to point at the real interpreter. Shebangs
    // are a Unix-only execution mechanism, so there is nothing to do on
    // Windows (where `.data/scripts/` content without a matching
    // `console_scripts`/`gui_scripts` entry point is not made executable by
    // any installer, rattler included).
    if let Some(python_info) = &python_info
        && platform.is_unix()
    {
        let target_prefix = target_prefix
            .to_str()
            .ok_or(InstallError::TargetPrefixIsNotUtf8)?;
        fix_up_script_shebangs(&wheel_record, target_dir, python_info, target_prefix)?;
    }

    Ok((paths, link_type))
}

/// Rewrites every `.data/scripts/` entry's placeholder shebang (see
/// [`rewrite_wheel_script_shebang`]) after linking.
fn fix_up_script_shebangs(
    wheel_record: &WheelRecord,
    target_dir: &Prefix,
    python_info: &PythonInfo,
    target_prefix: &str,
) -> Result<(), WheelInstallError> {
    for entry in &wheel_record.entries {
        if !is_wheel_script_path(&entry.archive_path) {
            continue;
        }
        let mapped = map_wheel_archive_path(&entry.archive_path);
        let destination = target_dir
            .path()
            .join(python_info.get_python_noarch_target_path(&mapped));
        rewrite_wheel_script_shebang(&destination, python_info, target_prefix)?;
    }
    Ok(())
}

/// Rewrites a `#!python`/`#!pythonw` placeholder shebang - the convention
/// [the wheel spec](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#script-wrapping)
/// uses for scripts in `<name>-<version>.data/scripts/` that installers are
/// expected to point at the *real* interpreter - to the environment's actual
/// Python executable. A script whose first line is anything else (already a
/// concrete interpreter path, or a `console_scripts`/`gui_scripts`-generated
/// launcher, which already has the right shebang - see
/// [`crate::install::entry_point`]) is left untouched.
///
/// Deliberately implemented as its own pass over the *destination* files
/// after linking, rather than by threading a `PrefixPlaceholder` through the
/// generic `link_file` machinery: that machinery's placeholder substitution
/// is a single, environment-prefix-wide find/replace keyed on
/// `InstallOptions::target_prefix`, not a per-file interpreter-path
/// substitution, so reusing it here would mean stuffing an unrelated meaning
/// into that field. Every affected file is rewritten via a fresh
/// temp-file-then-rename, never by editing in place, so a hardlinked
/// (shared-inode) cache entry is never mutated by this pass.
fn rewrite_wheel_script_shebang(
    path: &Path,
    python_info: &PythonInfo,
    target_prefix: &str,
) -> Result<(), WheelInstallError> {
    let content = match fs_err::read(path) {
        Ok(content) => content,
        // A symlink skipped during extraction (e.g. on Windows without the
        // right privileges) may legitimately be missing; nothing to fix up.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(WheelInstallError::Io(e)),
    };

    let first_line_end = content
        .iter()
        .position(|&b| b == b'\n')
        .unwrap_or(content.len());
    let first_line = content[..first_line_end]
        .strip_suffix(b"\r")
        .unwrap_or(&content[..first_line_end]);

    if first_line != b"#!python" && first_line != b"#!pythonw" {
        return Ok(());
    }

    let shebang = python_info.shebang(target_prefix);
    let mut new_content = Vec::with_capacity(shebang.len() + content.len() - first_line_end);
    new_content.extend_from_slice(shebang.as_bytes());
    new_content.extend_from_slice(&content[first_line_end..]);

    let permissions = fs_err::metadata(path)?.permissions();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut temp = tempfile::Builder::new()
        .prefix(".rattler-shebang-fixup")
        .tempfile_in(parent)
        .map_err(WheelInstallError::Io)?;
    temp.write_all(&new_content)
        .map_err(WheelInstallError::Io)?;
    temp.as_file()
        .set_permissions(permissions)
        .map_err(WheelInstallError::Io)?;
    temp.persist(path)
        .map_err(|e| WheelInstallError::Io(e.error))?;
    Ok(())
}

/// Links an already-cached (extracted) wheel from `cached_wheel_dir` into
/// `target_prefix`, writing a `PrefixRecord` to `conda-meta` in exactly the
/// same way as for a regular conda package. This is the wheel equivalent of
/// the internal `link_package` helper used by [`super::Installer`] for
/// conda packages.
///
/// Unlike the previous version of this function, this does *not* fetch or
/// extract the wheel itself: `cached_wheel_dir` must already be the
/// directory returned by [`populate_wheel_cache`] for this exact `record`.
/// Calling both in sequence for the same install used to fetch and extract
/// every wheel twice, into a nested cache directory - see
/// [`populate_wheel_cache`] and the module-level "Caching" section for how
/// the two are now kept as separate steps precisely to avoid that.
pub async fn install_wheel(
    record: &RepoDataRecord,
    target_prefix: &Prefix,
    cached_wheel_dir: &Path,
    install_options: InstallOptions,
    driver: &InstallDriver,
    requested_specs: Vec<String>,
) -> Result<(), WheelInstallError> {
    let record = record.clone();
    let target_prefix = target_prefix.clone();
    let clobber_registry = driver.clobber_registry.clone();
    let conda_meta_path = target_prefix.path().join("conda-meta");
    let cached_wheel_dir_for_task = cached_wheel_dir.to_path_buf();

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
    /// - a `.data/scripts/` file with a `#!python` placeholder shebang, as
    ///   real wheels ship per [the wheel
    ///   spec](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#script-wrapping)
    ///   -> should be rewritten to the real interpreter path on install
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
              demo-1.0.data/scripts/demo-shebang-script,sha256=,22\n\
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

        zip.start_file("demo-1.0.data/scripts/demo-shebang-script", options)
            .unwrap();
        zip.write_all(b"#!python\nprint('hi')\n").unwrap();

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
        // 6 files listed in `RECORD` plus the generated `demo-cli` entry
        // point script (1 file on unix, 2 on windows).
        let expected_paths = if cfg!(windows) { 8 } else { 7 };
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

        // The `#!python` placeholder shebang was rewritten to point at the
        // real interpreter (Unix only - shebangs are not an execution
        // mechanism on Windows).
        #[cfg(unix)]
        {
            let shebang_script_path = prefix_dir
                .join(&python_info.bin_dir)
                .join("demo-shebang-script");
            let content = std::fs::read_to_string(&shebang_script_path).unwrap();
            let first_line = content.lines().next().unwrap();
            assert_ne!(
                first_line, "#!python",
                "the placeholder shebang should have been rewritten"
            );
            assert!(
                first_line.starts_with("#!") && first_line.contains("python"),
                "expected a shebang pointing at a python interpreter, got: {first_line}"
            );
            assert!(content.ends_with("print('hi')\n"));
        }

        // The wheel cache holds exactly one, non-nested extraction per
        // wheel: a regression check for a bug where installing a wheel
        // fetched and extracted it a second time into a directory nested
        // inside the first.
        let wheel_cache_dir = temp_dir.path().join("wheel-cache");
        let mut extraction_dirs = std::fs::read_dir(&wheel_cache_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|t| t.is_dir()))
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        assert_eq!(
            extraction_dirs.len(),
            1,
            "expected exactly one top-level extraction directory in the wheel cache, found {extraction_dirs:?}"
        );
        let extraction_dir = extraction_dirs.pop().unwrap();
        assert_eq!(
            installed_record.extracted_package_dir.as_deref(),
            Some(extraction_dir.as_path()),
            "the installed record should point at the top-level cache extraction, not a nested copy"
        );
        // The cache holds a faithful, archive-relative unpack: `RECORD` is a
        // root-level entry, not remapped under `site-packages/`.
        assert!(extraction_dir.join("demo-1.0.dist-info/RECORD").is_file());
        assert!(!extraction_dir.join("site-packages").exists());

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

    /// A wheel whose repodata `sha256` does not match the archive actually
    /// served must fail to install, rather than being silently linked and
    /// tracked in `conda-meta` unverified (see the `WheelInstallError::FailedToFetch`
    /// path in `populate_wheel_cache`, which now goes through the same
    /// whole-archive hash verification a conda package fetch gets).
    ///
    /// This exercises a real HTTP fetch (a local server), not a `file://`
    /// URL: like conda's own `get_or_fetch_from_path`, a *local* wheel
    /// install intentionally does not hash-verify (see the module docs);
    /// only a URL fetch does.
    #[tokio::test]
    async fn test_wheel_hash_mismatch_fails_install() {
        use assert_matches::assert_matches;
        use std::future::IntoFuture;

        let temp_dir = tempfile::tempdir().unwrap();
        let wheel_path = temp_dir.path().join("demo-1.0-py3-none-any.whl");
        build_test_wheel(&wheel_path);

        let addr = std::net::SocketAddr::new([127, 0, 0, 1].into(), 0);
        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let service = axum::Router::new()
            .fallback_service(tower_http::services::ServeDir::new(temp_dir.path()))
            .into_make_service();
        tokio::spawn(axum::serve(listener, service).into_future());

        let prefix_dir = temp_dir.path().join("prefix");
        let prefix = Prefix::create(&prefix_dir).unwrap();

        let python_record = fake_python_repodata_record();
        let python_prefix_record =
            PrefixRecord::from_repodata_record(python_record.clone(), Vec::new());
        let conda_meta_dir = prefix_dir.join("conda-meta");
        std::fs::create_dir_all(&conda_meta_dir).unwrap();
        python_prefix_record
            .write_to_path(conda_meta_dir.join(python_prefix_record.file_name()), true)
            .unwrap();

        let mut wheel_record = wheel_repodata_record(&wheel_path);
        wheel_record.url = Url::parse(&format!("http://{addr}/demo-1.0-py3-none-any.whl")).unwrap();
        wheel_record.package_record.sha256 =
            rattler_digest::parse_digest_from_hex::<rattler_digest::Sha256>(&"0".repeat(64));

        let result = Installer::new()
            .with_wheel_cache_dir(temp_dir.path().join("wheel-cache"))
            .install(&prefix, vec![python_record, wheel_record])
            .await;

        assert_matches!(
            result,
            Err(crate::install::InstallerError::FailedToInstallWheel(_, _))
        );
        assert!(
            !prefix_dir
                .join("conda-meta")
                .join("demo-1.0-py3_none_any_0.json")
                .exists(),
            "a hash-mismatched wheel should not be tracked in conda-meta"
        );
    }
}
