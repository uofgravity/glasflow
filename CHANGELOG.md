# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0]

### Added

- Add various autoregressive flows using the existing transforms in `nflows` (https://github.com/uofgravity/glasflow/pull/62)
- Add `scale_activation` keyword argument to `nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform` (https://github.com/uofgravity/nflows/pull/11)

### Changed

- Drop support for Python 3.7 (https://github.com/uofgravity/glasflow/pull/61)


## [0.3.1]

### Fixed

- Addressed a deprecation warning in the `nflows` submodule when using LU decomposition (https://github.com/uofgravity/nflows/pull/10, https://github.com/uofgravity/glasflow/pull/57)

## [0.3.0]

### Added

- Keyword arguments passed to `glasflow.transform.coupling.AffineCouplingTransform` are now propogated to the parent class. ([#51](https://github.com/uofgravity/glasflow/pull/51))
- Add support `scale_activation` to `glasflow.transform.coupling.AffineCouplingTransform` and set the default to `nflows_general`. ([#52](https://github.com/uofgravity/glasflow/pull/52), [#54](https://github.com/uofgravity/glasflow/pull/54))

### Changed

- Default scale activation for `glasflow.transform.coupling.AffineCouplingTransform` is changed from `DEFAULT_SCALE_ACTIVATION` in nflows to `nflows_general` from glasflow. This changes the default behaviour, the previous behaviour can be recovered by setting `scale_activation='nflows'`. ([#52](https://github.com/uofgravity/glasflow/pull/52), [#54](https://github.com/uofgravity/glasflow/pull/54))

### Fixed

- fix a bug in `glasflow.nflows/utils/torchutils.searchsorted`, see https://github.com/uofgravity/nflows/pull/9 for details. ([#53](https://github.com/uofgravity/glasflow/pull/53))

### Deprecated

- The `scaling_method` argument in `glasflow.transform.coupling.AffineCouplingTransform` is now deprecated in favour of `scale_activation` and will be removed in a future release. ([#52](https://github.com/uofgravity/glasflow/pull/52))

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

[Unreleased]: https://github.com/uofgravity/glasflow/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/uofgravity/glasflow/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/uofgravity/glasflow/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/uofgravity/glasflow/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/uofgravity/glasflow/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/uofgravity/glasflow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/uofgravity/glasflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/uofgravity/glasflow/releases/tag/v0.1.0
