# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2]

Another patch to fix CI not uploading release to PyPI
### Changed

- Update `nflows` submodule ([#36](https://github.com/igr-ml/glasflow/pull/36))
- Remove LFS from `publish-to-pypi` workflow ([#36](https://github.com/igr-ml/glasflow/pull/36))

## [0.1.1]

Patch to fix CI not uploading release to PyPI

### Changed

- Add LFS to `publish-to-pypi` workflow  ([#35](https://github.com/igr-ml/glasflow/pull/35))

## [0.1.0]

### Added

- Add `RealNVP`
- Add `CouplingNSF` (Coupling Neural Spline Flow)
- Add `nflows` submodule that replaces `nflows` dependency
- Add option for user-defined masks in coupling-based flows

[Unreleased]: https://github.com/igr-ml/glasflow/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/igr-ml/glasflow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/igr-ml/glasflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/igr-ml/glasflow/releases/tag/v0.1.0
