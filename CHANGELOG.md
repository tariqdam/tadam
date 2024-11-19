# Changelog

## [Unreleased]

### Added
- `load` and `dump` functions which automatically compresses `gzip` at level 6
- `dataprocessing.merge_windows_with_measurements` now uses `duckdb` instead of 
  `pandas` for faster merging using less memory
- `assertions` module
- `assertions.assert_stop_after_start` to check if the data is validly structured
- `utils.load` and `utils.dump` to load and dump hashed files

### Planned
- Add logger method to log to file and console

## [0.0.1] - 2023-07-10

### Added

- Initial release
- License set to MIT
- Pre-commit hooks for code formatting and linting
- Poetry for dependency management
- Data processing functions
- Unit tests for parts of functions

### Changed

- Repository location and package name

### Removed

- Unused files and folders from previous repository

[unreleased]: https://github.com/tariqdam/tadam/-/compare/0.0.1...HEAD
[0.0.1]: https://github.com/tariqdam/tadam/-/releases/tag/0.0.1