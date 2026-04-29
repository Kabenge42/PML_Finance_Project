# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.6] - 2026-04-16

### Changed

- Migrated to ArviZ 1.0: fixed `az.from_dict()` usage across the package, updated `Pipfile` dependencies, and resolved
  compatibility bugs ([`3a06d8c`](https://github.com/Kabenge42/PML_Finance_Project/commit/3a06d8c))
- Migrated analytics modules to the new `probabilistic_ml_model` package ([
  `b157963`](https://github.com/Kabenge42/PML_Finance_Project/commit/b157963))

### Removed

- Deprecated inspection profiles and statistical-functions modules from `probabilistic_ml_model` ([
  `fec6015`](https://github.com/Kabenge42/PML_Finance_Project/commit/fec6015))

### Added

- Expected-returns analytics log file for pipeline tracking ([
  `b157963`](https://github.com/Kabenge42/PML_Finance_Project/commit/b157963))

[0.9.6]: https://github.com/Kabenge42/PML_Finance_Project/compare/v0.9.5...v0.9.6
