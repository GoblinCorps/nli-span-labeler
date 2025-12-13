# NLI Span Labeler - GoblinCorps Development Guide

## Team Roles

### PR Workflow

| Role | Goblin | Responsibilities |
|------|--------|------------------|
| **PR Review** | Frick | Reviews PRs, requests changes, resolves merge conflicts, adds fix commits |
| **Merge & Test** | Contraption | Runs tests, merges approved PRs, rejects if tests fail |
| **Feature Development** | Frack | Implements features, bug fixes, new functionality |

### PR Process

1. **Submission**: Developer opens PR or external contributor submits changes
2. **Review (Frick)**:
   - Reviews code for correctness and adherence to project standards
   - Requests changes or adds fix commits as needed
   - Resolves merge conflicts with main branch
   - Approves PR when ready
3. **Merge (Contraption)**:
   - Runs full test suite on approved PRs
   - If tests pass → Merge PR
   - If tests fail with trivial fixes → Fix and merge
   - If tests fail with non-trivial issues → Reject PR back to reviewer

### Test Maintenance

Contraption is responsible for:
- Keeping the test suite passing and up to date
- Adding tests for new functionality
- Fixing broken tests after merges
- Maintaining test coverage standards

## Development Standards

### Commit Messages
- Use conventional commit format when appropriate
- Be descriptive about what changed and why

### Testing
- All PRs must pass existing tests before merge
- New features should include corresponding tests
- Test failures block merging

## Getting Help

- **User coordination & issues**: @Frick
- **Feature implementation**: @Frack  
- **Tests, docs, infrastructure**: @Contraption
