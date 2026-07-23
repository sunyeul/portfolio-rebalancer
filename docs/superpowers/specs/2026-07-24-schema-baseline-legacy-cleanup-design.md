# Schema Baseline and Legacy Cleanup Design

**Status:** Proposed

## Goal

Remove the retired manual instrument-metadata model and its historical schema
implementation from the repository. Keep the current Toss-only v10 database
schema as the sole supported baseline.

## Scope

- Remove historical migration SQL and migration-version tests, including the
  retired instrument-profile cleanup migration.
- Build fresh databases directly from the complete current v10 schema.
- Keep an existing v10 database readable without schema mutation.
- Reject databases below v10 without mutation or automatic deletion.
- Remove retired metadata references from agent guidance and obsolete design
  documentation.

The retired model includes instrument profiles and their thesis, overlap,
burden, holdability, ETF-substitution, and review-factor fields.

## Compatibility Contract

`data/portfolio_rebalancer.sqlite3` is currently v10 and contains no
`ips_instrument_profiles` table. It remains supported as-is.

Databases with versions 1 through 9 are no longer upgradeable. Startup must
fail closed with a machine-readable error that identifies the unsupported
version and instructs the operator to create a new database explicitly. The
application must not delete, rewrite, or silently migrate an old database.

Databases newer than v10 remain rejected without mutation.

## Schema Design

Replace the numbered migration map with one current-schema SQL definition.
Database initialization has three paths:

1. An absent or empty database receives the current schema and is stamped with
   `PRAGMA user_version = 10` atomically.
2. A v10 database receives only integrity and foreign-key validation.
3. Any nonzero version below or above v10 raises `SchemaVersionError` before
   schema writes occur.

The current schema definition must contain only active Toss snapshots,
policies, performance, market evidence, and evaluation tables. It must not
mention profile tables, profile hashes, profile snapshots, or manual
instrument-review fields.

## Runtime and Guidance Cleanup

Remove the retired `thesis_status=broken` condition from the IPS judgment
guidance. In the active model, no user-facing or persisted thesis-status input
exists; inspection output must not imply otherwise.

Delete the retirement design document after its requirements have been folded
into this baseline design and implementation tests. Keep no runtime, API,
agent-guidance, or test references to the retired metadata fields. This design
is the sole retained decision record for the cleanup.

## Verification

- A fresh database has schema version 10, passes integrity checks, and contains
  only active tables and columns.
- A copied v10 database opens without schema changes and preserves its Toss
  snapshots, policies, performance runs, market evidence, and evaluations.
- A v1 through v9 fixture is rejected without table, data, or version changes.
- A future-version fixture is rejected without mutation.
- A repository-wide hidden-file search of runtime, API, agent-guidance, and
  test paths finds no retired metadata identifiers.
- Run the focused schema, database, inspection, CLI, and API suites, then the
  full Python test suite.

## Non-goals

- No portfolio holdings, policies, snapshots, or user state are deleted.
- No market sync, investment recommendation, trade, or order preparation is
  introduced.
