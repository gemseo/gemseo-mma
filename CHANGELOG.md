<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Changelog titles are:
- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.
-->

# Changelog

All notable changes of this project will be documented here.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0)
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Changed

- The `MMASvanbergSettings` class was moved to `gemseo_mma.opt.settings.mma_settings` and renamed
  to `MMASvanberg_Settings` in order to follow GEMSEO conventions.


## Version 3.0.0 (November 2024)

### Added

- Support GEMSEO v6.
- Support for Python 3.12.

## Version 2.0.2 (August 2024)

### Fixed

- The license of the package is now GPLv3 since it uses a module that is GPLv3.

## Version 2.0.1 (December 2023)

### Added

- Support for Python 3.11.

### Fixed

- A bug on the option handling was solved for design space normalization
  and inequality constraint tolerance.

### Removed

- Support for Python 3.8.

## Version 2.0.0 (June 2023)

Update to GEMSEO 5.

### Fixed

- A bug on the option settings was solved.

### Changed

- The `ctol_abs` option was removed, this was anyway not used.
- The attributes and option names were changed to be more explicit.
- The solver attributes are made private.

## Version 1.0.0 (February 2023)

First release.
