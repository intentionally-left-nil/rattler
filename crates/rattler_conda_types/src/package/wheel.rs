//! Support for reading metadata out of an *extracted* Python wheel (`.whl`)
//! archive.
//!
//! A wheel is a plain zip file. Its own internal layout follows [the wheel
//! specification](https://packaging.python.org/en/latest/specifications/binary-distribution-format/):
//!
//! * Most files live at the root of the archive and are installed into the
//!   environment's `site-packages` directory (this includes the
//!   `<name>-<version>.dist-info/` directory itself, which every RECORD
//!   naturally lists).
//! * A `<name>-<version>.data/` directory may contain sub-directories named
//!   `purelib`, `platlib`, `scripts`, `headers` or `data` whose *contents*
//!   should be installed relative to some other location than
//!   `site-packages` (respectively: `site-packages` again, `site-packages`
//!   again, the environment's script/binary directory, an `include`
//!   directory, and the environment root).
//!
//! [`map_wheel_archive_path`] implements this remapping. It is used while
//! turning the wheel's `RECORD` file into a rattler
//! [`crate::prefix_record::PathsEntry`]-like manifest (to decide the
//! logical, noarch-python-style destination path of each installed file,
//! see [`crate::package::PathsEntry`] and the `site-packages/`/
//! `python-scripts/` prefix convention used for `noarch: python` conda
//! packages).
//!
//! Deliberately, this remapping is *not* applied while extracting the wheel
//! archive itself: the package cache holds a faithful, archive-relative
//! unpack of the wheel (the same layout `unzip` would produce), and the
//! `site-packages`/`python-scripts` remapping is applied only when computing
//! each file's install-time destination. This keeps the on-disk cache format
//! independent of this remapping convention, so that a future change to the
//! convention (e.g. a more faithful `headers` mapping) does not silently
//! invalidate or corrupt existing cache entries, and so that the wheel's own
//! `RECORD` hashes can be validated directly against the cached files without
//! having to reverse the remapping first.
//!
//! This module intentionally only deals with information that must be read
//! from the wheel's own contents: the file manifest (`RECORD`) and console
//! script entry points (`entry_points.txt`). The package name, version, build
//! string and dependencies are already known from the (v3) repodata record
//! that pointed at the wheel and do not need to be re-derived here.

use std::{
    io,
    path::{Component, Path, PathBuf},
};

use base64::Engine;
use rattler_digest::Sha256Hash;

/// The category of a path inside a wheel's `<name>-<version>.data/`
/// directory, as defined by [the wheel
/// specification](https://packaging.python.org/en/latest/specifications/binary-distribution-format/).
/// `None` (returned by [`wheel_archive_category`]) means the path is not
/// inside a `*.data/` directory at all, i.e. it is a "root" wheel file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WheelDataCategory {
    /// `<name>-<version>.data/purelib/` - installed into `site-packages/`.
    Purelib,
    /// `<name>-<version>.data/platlib/` - installed into `site-packages/`.
    Platlib,
    /// `<name>-<version>.data/scripts/` - installed into the environment's
    /// script/binary directory.
    Scripts,
    /// `<name>-<version>.data/headers/` - installed as-is, relative to the
    /// environment root (see [`map_wheel_archive_path`] for the caveat about
    /// this simplification).
    Headers,
    /// `<name>-<version>.data/data/`, or any future/unrecognized category -
    /// installed as-is, relative to the environment root.
    Data,
}

/// Determines whether `path` (as it appears inside a wheel archive, or in a
/// wheel's `RECORD` file) is inside a `<name>-<version>.data/<category>/`
/// directory, and if so, returns the category together with the path
/// relative to that category directory.
pub fn wheel_archive_category(path: &Path) -> Option<(WheelDataCategory, PathBuf)> {
    let components: Vec<Component<'_>> = path.components().collect();

    let Some(Component::Normal(first)) = components.first() else {
        return None;
    };
    if !first.to_string_lossy().ends_with(".data") || components.len() < 2 {
        return None;
    }
    let Component::Normal(category) = &components[1] else {
        return None;
    };

    let rest: PathBuf = components[2..].iter().collect();
    let category = match category.to_string_lossy().as_ref() {
        "purelib" => WheelDataCategory::Purelib,
        "platlib" => WheelDataCategory::Platlib,
        "scripts" => WheelDataCategory::Scripts,
        "headers" => WheelDataCategory::Headers,
        // "data", or any future/unknown category.
        _ => WheelDataCategory::Data,
    };
    Some((category, rest))
}

/// Maps a path as it appears inside a wheel archive (or a wheel's `RECORD`
/// file) onto the logical, install-location-relative path that rattler uses
/// for `noarch: python` conda packages.
///
/// * Paths inside a `<name>-<version>.data/purelib/` or
///   `.../platlib/` directory are mapped into `site-packages/`.
/// * Paths inside a `.../scripts/` directory are mapped into
///   `python-scripts/` (rattler's convention for the environment's binary
///   directory, see [`crate::package::PathType`] handling for noarch python
///   packages).
/// * Paths inside a `.../headers/` or `.../data/` directory (or any other,
///   unrecognized, category) are mapped as-is, relative to the environment
///   root. This is a pragmatic simplification: these categories are rarely
///   used in practice and, unlike `purelib`/`platlib`/`scripts`, are not
///   Python-version-specific, so a literal environment-root-relative path is
///   a reasonable approximation of pip's own behavior for the `data`
///   category (pip's `headers` handling is more involved, placing files
///   under a Python-version-specific `include` directory; that nuance is not
///   replicated here).
/// * Every other path (i.e. everything that is not inside a `*.data/`
///   directory) is a "root" wheel file and is mapped into `site-packages/`.
///   This covers both `purelib` and `platlib` wheels uniformly, which is
///   correct for virtually all Python installations where both concepts
///   resolve to the same `site-packages` directory.
pub fn map_wheel_archive_path(path: &Path) -> PathBuf {
    if let Some((category, rest)) = wheel_archive_category(path) {
        return match category {
            WheelDataCategory::Purelib | WheelDataCategory::Platlib => {
                Path::new("site-packages").join(rest)
            }
            WheelDataCategory::Scripts => Path::new("python-scripts").join(rest),
            WheelDataCategory::Headers | WheelDataCategory::Data => rest,
        };
    }

    Path::new("site-packages").join(path)
}

/// Returns `true` if `path` (a wheel archive path) is installed into the
/// environment's script/binary directory, i.e. maps under
/// `python-scripts/` (see [`map_wheel_archive_path`]).
pub fn is_wheel_script_path(path: &Path) -> bool {
    matches!(
        wheel_archive_category(path),
        Some((WheelDataCategory::Scripts, _))
    )
}

/// A single entry of a wheel's `RECORD` file.
///
/// See [the wheel specification](https://packaging.python.org/en/latest/specifications/recording-installed-packages/)
/// for the exact format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WheelRecordEntry {
    /// The path of the file as it appears in the wheel archive (i.e. relative
    /// to the root of the archive, *not* remapped to its install location).
    pub archive_path: PathBuf,

    /// The (optional) sha256 hash of the file, as recorded in `RECORD`. This
    /// is `None` if the `RECORD` file did not specify a hash for this entry
    /// (this is common for the `RECORD` file's own entry, since its hash
    /// cannot be known before the file itself is finalized) or if the hash
    /// used an algorithm other than sha256.
    pub sha256: Option<Sha256Hash>,

    /// The (optional) size of the file in bytes, as recorded in `RECORD`.
    pub size: Option<u64>,
}

impl WheelRecordEntry {
    /// Returns the logical, install-location-relative path of this entry. See
    /// [`map_wheel_archive_path`].
    pub fn mapped_path(&self) -> PathBuf {
        map_wheel_archive_path(&self.archive_path)
    }
}

/// The parsed contents of a wheel's `<name>-<version>.dist-info/RECORD` file:
/// a manifest of every file the wheel contains.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WheelRecord {
    /// The entries listed in the `RECORD` file.
    pub entries: Vec<WheelRecordEntry>,
}

impl WheelRecord {
    /// Parses a `WheelRecord` from the raw contents of a `RECORD` file.
    ///
    /// `RECORD` is a CSV-like file (see [PEP 376](https://peps.python.org/pep-0376/))
    /// with one `<path>,<hash>,<size>` entry per line. `<hash>` has the form
    /// `sha256=<url-safe base64, no padding>` or is empty. `<size>` is empty
    /// or a decimal number. Empty lines are ignored.
    pub fn parse(content: &str) -> io::Result<Self> {
        let mut entries = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let fields = split_record_line(line);
            let Some(path) = fields.first() else {
                continue;
            };
            if path.is_empty() {
                continue;
            }

            let sha256 = fields
                .get(1)
                .filter(|hash| !hash.is_empty())
                .and_then(|hash| parse_record_hash(hash));
            let size = fields
                .get(2)
                .filter(|size| !size.is_empty())
                .and_then(|size| size.parse::<u64>().ok());

            entries.push(WheelRecordEntry {
                archive_path: PathBuf::from(path),
                sha256,
                size,
            });
        }
        Ok(Self { entries })
    }

    /// Reads and parses the `RECORD` file that was extracted from a wheel
    /// archive into `extracted_wheel_dir`.
    ///
    /// See [`find_dist_info_dir`] for how the `.dist-info` directory is
    /// located.
    pub fn from_extracted_directory(extracted_wheel_dir: &Path) -> io::Result<Self> {
        let dist_info_dir = find_dist_info_dir(extracted_wheel_dir)?;
        let record_path = dist_info_dir.join("RECORD");
        let content = fs_err::read_to_string(&record_path)?;
        Self::parse(&content)
    }
}

/// Locates the single `<name>-<version>.dist-info` directory inside an
/// extracted wheel directory.
///
/// The extracted wheel directory holds a faithful, archive-relative unpack
/// of the wheel (i.e. the same layout `unzip` would produce, *not* remapped
/// via [`map_wheel_archive_path`] - see the module-level documentation), so
/// the `.dist-info` directory (a root-level entry in every wheel archive) is
/// looked for directly under `extracted_wheel_dir`.
pub fn find_dist_info_dir(extracted_wheel_dir: &Path) -> io::Result<PathBuf> {
    let mut found = None;
    for entry in fs_err::read_dir(extracted_wheel_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let name = entry.file_name();
            if name.to_string_lossy().ends_with(".dist-info") {
                if found.is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "found multiple '*.dist-info' directories in wheel archive",
                    ));
                }
                found = Some(entry.path());
            }
        }
    }
    found.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "could not find a '*.dist-info' directory in wheel archive",
        )
    })
}

/// Parses the `[console_scripts]` section of a wheel's `entry_points.txt`
/// file (see the [entry points
/// specification](https://packaging.python.org/en/latest/specifications/entry-points/)),
/// returning the raw, unparsed `name = module:function` lines.
///
/// Any trailing `[extra1,extra2]` marker (used for optional dependencies, not
/// applicable to console script entry points but tolerated here for
/// robustness) is stripped before the line is returned.
pub fn parse_console_scripts(entry_points_txt: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut in_console_scripts = false;
    for line in entry_points_txt.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_console_scripts = trimmed.eq_ignore_ascii_case("[console_scripts]");
            continue;
        }
        if in_console_scripts {
            let line = match trimmed.find('[') {
                Some(idx) => trimmed[..idx].trim_end(),
                None => trimmed,
            };
            if !line.is_empty() {
                result.push(line.to_string());
            }
        }
    }
    result
}

/// Splits a single line of a `RECORD` file into its `,`-separated fields,
/// honoring `"`-quoted fields (with `""` as an escaped quote) as used by
/// `csv.writer` (which `pip`/`installer` use to generate `RECORD`).
fn split_record_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            '"' if !in_quotes && current.is_empty() => {
                in_quotes = true;
            }
            ',' if !in_quotes => {
                fields.push(std::mem::take(&mut current));
            }
            c => current.push(c),
        }
    }
    fields.push(current);
    fields
}

/// Parses a `RECORD` hash field (e.g. `sha256=<url-safe base64, no
/// padding>`) into a [`Sha256Hash`]. Returns `None` if the field does not use
/// the `sha256` algorithm or could not be parsed.
fn parse_record_hash(field: &str) -> Option<Sha256Hash> {
    let value = field.strip_prefix("sha256=")?;
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(value)
        .ok()?;
    let hex_str = hex::encode(decoded);
    rattler_digest::parse_digest_from_hex::<rattler_digest::Sha256>(&hex_str)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_map_wheel_archive_path_root_file() {
        assert_eq!(
            map_wheel_archive_path(Path::new("six.py")),
            Path::new("site-packages/six.py")
        );
        assert_eq!(
            map_wheel_archive_path(Path::new("six-1.9.0.dist-info/RECORD")),
            Path::new("site-packages/six-1.9.0.dist-info/RECORD")
        );
    }

    #[test]
    fn test_map_wheel_archive_path_data_purelib_and_platlib() {
        assert_eq!(
            map_wheel_archive_path(Path::new("foo-1.0.data/purelib/foo/bar.py")),
            Path::new("site-packages/foo/bar.py")
        );
        assert_eq!(
            map_wheel_archive_path(Path::new("foo-1.0.data/platlib/foo/_native.so")),
            Path::new("site-packages/foo/_native.so")
        );
    }

    #[test]
    fn test_map_wheel_archive_path_data_scripts() {
        assert_eq!(
            map_wheel_archive_path(Path::new("foo-1.0.data/scripts/foo-cli")),
            Path::new("python-scripts/foo-cli")
        );
    }

    #[test]
    fn test_map_wheel_archive_path_data_data_and_headers() {
        assert_eq!(
            map_wheel_archive_path(Path::new("foo-1.0.data/data/share/foo/config.ini")),
            Path::new("share/foo/config.ini")
        );
        assert_eq!(
            map_wheel_archive_path(Path::new("foo-1.0.data/headers/foo.h")),
            Path::new("foo.h")
        );
    }

    #[test]
    fn test_is_wheel_script_path() {
        assert!(is_wheel_script_path(Path::new(
            "foo-1.0.data/scripts/foo-cli"
        )));
        assert!(!is_wheel_script_path(Path::new(
            "foo-1.0.data/purelib/foo/bar.py"
        )));
        assert!(!is_wheel_script_path(Path::new("foo.py")));
    }

    #[test]
    fn test_find_dist_info_dir_looks_at_archive_root() {
        let temp_dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(temp_dir.path().join("foo-1.0.dist-info")).unwrap();
        std::fs::write(temp_dir.path().join("foo.py"), "").unwrap();

        let found = find_dist_info_dir(temp_dir.path()).unwrap();
        assert_eq!(found, temp_dir.path().join("foo-1.0.dist-info"));
    }

    #[test]
    fn test_wheel_record_parse() {
        let record = "six.py,sha256=oVfM8HHRJHtSpvT6H9nCwXlH0Br1nhIQyfmAvxSHu9c,34547\n\
             six-1.9.0.dist-info/RECORD,,\n\
             \n";
        let parsed = WheelRecord::parse(record).unwrap();
        assert_eq!(parsed.entries.len(), 2);
        assert_eq!(parsed.entries[0].archive_path, Path::new("six.py"));
        assert!(parsed.entries[0].sha256.is_some());
        assert_eq!(parsed.entries[0].size, Some(34547));
        assert_eq!(
            parsed.entries[1].archive_path,
            Path::new("six-1.9.0.dist-info/RECORD")
        );
        assert_eq!(parsed.entries[1].sha256, None);
        assert_eq!(parsed.entries[1].size, None);
    }

    #[test]
    fn test_parse_console_scripts() {
        let entry_points = "[console_scripts]\n\
             foo = foo.cli:main\n\
             bar = foo.cli:bar_main\n\
             \n\
             [some.other.section]\n\
             baz = foo.other:baz\n";
        let scripts = parse_console_scripts(entry_points);
        assert_eq!(
            scripts,
            vec![
                "foo = foo.cli:main".to_string(),
                "bar = foo.cli:bar_main".to_string(),
            ]
        );
    }

    #[test]
    fn test_parse_console_scripts_strips_extras_marker() {
        let entry_points = "[console_scripts]\nfoo = foo.cli:main [extra]\n";
        let scripts = parse_console_scripts(entry_points);
        assert_eq!(scripts, vec!["foo = foo.cli:main".to_string()]);
    }
}
