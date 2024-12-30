# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Switch from `distutils` (deprecated) to `packaging`
  [[PR](https://github.com/f-dangel/unfoldNd/pull/39)]

## [0.2.2] - 2024-06-14

- Switch from `pkg_resources` (deprecated) to `importlib`+`distutils`, deprecate
  Python 3.7
  [[PR](https://github.com/f-dangel/unfoldNd/pull/37)]

## [0.2.1] - 2024-01-10

### Fixed

- Bug for non-zero padding in fold
  [[bug report](https://github.com/f-dangel/unfoldNd/issues/30)]
  [[PR](https://github.com/f-dangel/unfoldNd/pull/21)]

## [0.2.0] - 2022-11-10

### Added

- Generalization of `im2col` to transpose convolution
  [[PR](https://github.com/f-dangel/unfoldNd/pull/27)]

- Deprecate python 3.6
  [[PR](https://github.com/f-dangel/unfoldNd/pull/26)]


## [0.1.0] - 2021-05-05

### Added

- Generalization of `torch.nn.Fold`
  [[PR](https://github.com/f-dangel/unfoldNd/pull/18)]

### Fixed

- Bug related to data type of input to `unfoldNd`
  [[PR](https://github.com/f-dangel/unfoldNd/pull/21)]

### Internal

- Move to `setup.cfg`, automate PyPI release
  [[PR](https://github.com/f-dangel/unfoldNd/pull/22)]
- Import sorting
  [[PR](https://github.com/f-dangel/unfoldNd/pull/19)]

## [0.0.1] - 2021-02-19

Initial release

[Unreleased]: https://github.com/f-dangel/unfoldNd/compare/0.2.2...HEAD
[0.2.2]: https://github.com/f-dangel/unfoldNd/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/f-dangel/unfoldNd/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/f-dangel/unfoldNd/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/f-dangel/unfoldNd/compare/0.0.1...0.1.0
[0.0.1]: https://github.com/f-dangel/unfoldNd/releases/tag/0.0.1
