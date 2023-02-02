# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]

### Added

- Add a multi-layer perceptron (`glasflow.nets.mlp.MLP`). ([#40](https://github.com/uofgravity/glasflow/pull/40))
- Add a resampled Gaussian distribution that uses Learnt Accept/Reject Sampling (`glasflow.distributions.resampled.ResampledGaussian`). ([#40](https://github.com/uofgravity/glasflow/pull/40))
- Add `nessai.utils.get_torch_size`. ([#40](https://github.com/uofgravity/glasflow/pull/40))
- Add a multivariate uniform distribution for Neural Spline Flows (`glasflow.distributions.uniform.MultivariateUniform`). ([#47](https://github.com/uofgravity/glasflow/pull/47))

## Changed

- Change logging statements on import to, by default, only appear when an external version of nflows is being used. ([#44](https://github.com/uofgravity/glasflow/pull/44))

## [0.1.2]

Another patch to fix CI not uploading release to PyPI
### Changed

- Update `nflows` submodule ([#36](https://github.com/uofgravity/glasflow/pull/36))
- Remove LFS from `publish-to-pypi` workflow ([#36](https://github.com/uofgravity/glasflow/pull/36))

## [0.1.1]

Patch to fix CI not uploading release to PyPI

### Changed

- Add LFS to `publish-to-pypi` workflow  ([#35](https://github.com/uofgravity/glasflow/pull/35))

## [0.1.0]

### Added

- Add `RealNVP`
- Add `CouplingNSF` (Coupling Neural Spline Flow)
- Add `nflows` submodule that replaces `nflows` dependency
- Add option for user-defined masks in coupling-based flows

[Unreleased]: https://github.com/uofgravity/glasflow/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/uofgravity/glasflow/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/uofgravity/glasflow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/uofgravity/glasflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/uofgravity/glasflow/releases/tag/v0.1.0
