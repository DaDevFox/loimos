# Basic Chares: Primality Testing (Scaffolding)

This directory contains the initial scaffolding for implementing the "Basic Chares: Primality Testing" exercises. The code currently focuses on setting up the Charm++ module, entry methods, and build system so that the detailed logic for parts A–C can be filled in later.

## Layout

- `primality.ci` — Charm++ interface file defining the main chare, worker chare, and message record types used for single-number and batched primality tests.
- `Primality.C` — C++ implementation stub featuring argument parsing, result bookkeeping structures, and placeholder entry method bodies.
- `Makefile` — Minimal build script that relies on `charmc` being available on the `PATH`.

## Next Steps

1. Implement the task dispatch logic in `Main::startComputation` so that the configured number of tasks is issued immediately without array searches on result callbacks.
2. Fill in the `CheckPrimality` entry methods with the actual primality checks (including the optimizations requested for extra credit) and invoke the appropriate callback to deliver results to `Main::receiveResult`.
3. Extend the scaffolding to support grainsize control, timing runs, and 64-bit number generation as required by parts B and C.

The current scaffolding exits immediately after printing configuration information, making it safe to compile while the remaining functionality is being developed.
