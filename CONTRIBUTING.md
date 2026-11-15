# Contributing to ForgeOCR

Contributions are welcome: bug reports, benchmarks on new kinds of documents, fixes and features.

## Before your first pull request

ForgeOCR is owned by Znoow and released under Apache-2.0. To keep the licensing manageable,
every contributor accepts the [Contributor License Agreement](CLA.md) once. You keep the
copyright of what you write; you grant the licenses the CLA describes, which include the right
to also offer the Project under other terms, commercial ones included.

To accept it, keep this line in your pull request description (the template includes it):

> I have read the CLA and I agree to its terms.

Pull requests without it can't be merged.

## How to propose a change

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Keep each pull request to one change, with tests (`cargo test --release`).
3. For changes that touch speed or accuracy, include numbers from `bench/` before and after.
