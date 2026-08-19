# Release checklist example

## Identity

- [ ] Release version:
- [ ] Source commit SHA:
- [ ] Annotated tag:
- [ ] CI run:
- [ ] Artifact digest:

## Validation

- [ ] Required checks passed for the release SHA or merge group.
- [ ] Dependency lock and toolchain versions are recorded.
- [ ] SBOM and provenance/attestation were generated.
- [ ] The same artifact digest passed staging verification.
- [ ] Migration and backward compatibility were reviewed.

## Delivery

- [ ] Production environment approval is complete.
- [ ] Deployment concurrency prevents unsafe overlap.
- [ ] Rollout stages and stop conditions are defined.
- [ ] Known-good rollback digest is recorded.
- [ ] Metrics, logs, and alerts for verification are ready.

## Completion

- [ ] Release notes include breaking changes and operator actions.
- [ ] Deployment record links environment to artifact digest.
- [ ] Temporary flags or mitigations have owners and removal dates.
- [ ] Post-release verification is complete.
