# Git Conventions and Workflow

## Overview

This document defines the git workflow conventions and automated commit practices for the AI Hydra project. These conventions ensure consistent version control practices and automated change tracking.

## Automated Git Commits

### AI Agent File Modification Protocol

When an AI agent modifies files during development, the following automated git commit protocol must be followed:

#### 1. Automatic Commit Requirement
- **MANDATORY**: Every file modification by an AI agent must result in an automatic git commit
- **TIMING**: Commits should be made immediately after file modifications are complete
- **SCOPE**: Each logical change or set of related changes should be committed together

#### 2. Commit Message Format

**Standard Format:**
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**MANDATORY COMMIT CATEGORIES:**

All commits MUST use one of the following standardized categories:

| Category   | Description |
|------------|-------------|
| `feat`     | A new feature. |
| `fix`      | A bug fix. |
| `docs`     | Documentation only changes. |
| `style`    | Formatting changes, not code changes. |
| `refactor` | Code changes that neither fixes a bug nor adds a feature. |
| `test`     | Adding or refactoring Tests. |
| `chore`    | Build process or auxiliary tool changes. |

**Category Selection Guidelines:**
- Use `feat` for any new functionality, components, or capabilities
- Use `fix` for bug fixes, error corrections, or issue resolutions
- Use `docs` for documentation-only changes (README, comments, guides)
- Use `style` for code formatting, whitespace, or style-only changes
- Use `refactor` for code restructuring without changing functionality
- Use `test` for adding, modifying, or fixing tests
- Use `chore` for build scripts, dependencies, or maintenance tasks

**Examples:**
```bash
# Feature implementation
git commit -m "feat(simulation): implement neural network integration in pipeline

- Added NN components to SimulationPipeline
- Integrated feature extraction and oracle training
- Updated component statistics collection"

# Bug fix
git commit -m "fix(tests): resolve failing pipeline integration test

- Modified _collect_component_statistics() to use HydraMgr components
- Added backward compatibility for master game statistics
- Test now passes successfully"

# Documentation update
git commit -m "docs(steering): add git conventions for automated commits

- Created git-conventions.md with AI agent commit protocol
- Defined commit message format and examples
- Established file modification workflow"
```

#### 3. Commit Content Guidelines

**What to Include:**
- All modified files related to the current task
- Generated or updated test files
- Documentation updates that accompany code changes
- Configuration changes that support the modifications

**What to Exclude:**
- Temporary files or build artifacts
- IDE-specific configuration files
- Files unrelated to the current change

#### 4. Implementation Requirements

**For AI Agents:**
1. **Detect Changes**: Monitor which files have been modified during the session
2. **Generate Message**: Create descriptive commit message based on changes made
3. **Stage Files**: Add all relevant modified files to git staging
4. **Commit**: Execute git commit with the generated message
5. **Verify**: Confirm commit was successful

**Example Implementation Flow:**
```bash
# 1. Stage modified files
git add ai_hydra/simulation_pipeline.py ai_hydra/master_game.py

# 2. Commit with descriptive message
git commit -m "fix(pipeline): resolve component integration issue

- Modified SimulationPipeline._collect_component_statistics()
- Now uses HydraMgr components instead of separate instances
- Added backward compatibility for master_game statistics
- Fixes test_pipeline_component_integration test failure"

# 3. Verify commit
git log --oneline -1
```

## Manual Git Workflow

### Branch Management

**Main Branch:**
- `main`: Production-ready code
- All commits should be functional and tested
- Direct commits allowed for small fixes and documentation

**Feature Branches:**
- Use for major feature development
- Format: `feature/description-of-feature`
- Merge back to main when complete

### Commit Best Practices

#### 1. Atomic Commits
- Each commit should represent a single logical change
- Avoid mixing unrelated changes in one commit
- Make commits that can be easily reverted if needed

#### 2. Descriptive Messages
- First line: concise summary (50 characters or less)
- Body: detailed explanation if needed (wrap at 72 characters)
- Reference issue numbers when applicable

#### 3. Commit Frequency
- Commit early and often
- Don't wait until end of day to commit changes
- Each working feature or fix should be committed

### Code Review Process

#### 1. Self Review
- Review your own changes before committing
- Check for debugging code, console.log statements, etc.
- Ensure tests pass locally

#### 2. Automated Checks
- All commits trigger automated testing
- Documentation builds must succeed
- Code style checks must pass

## Git Hooks and Automation

### Pre-commit Hooks
- Code formatting (Black, isort)
- Linting (flake8, mypy)
- Test execution for modified components
- Documentation syntax validation

### Post-commit Actions
- Automated testing pipeline
- Documentation rebuilding
- Coverage report generation

### Auto Git Commit Hook
- Monitors file changes in the workspace
- Automatically commits changes when files are saved
- Uses intelligent commit message generation
- Can be configured in Kiro IDE settings

## File-Specific Commit Patterns

### Code Files
```bash
# New implementation
git commit -m "feat(component): implement new functionality"

# Bug fix
git commit -m "fix(component): resolve specific issue"

# Refactoring
git commit -m "refactor(component): improve code structure"
```

### Test Files
```bash
# New tests
git commit -m "test(component): add comprehensive test coverage"

# Fix failing tests
git commit -m "fix(tests): resolve test failures in component"

# Property-based tests
git commit -m "test(pbt): add property-based tests for component"
```

### Documentation Files
```bash
# New documentation
git commit -m "docs(topic): add comprehensive documentation"

# Update existing docs
git commit -m "docs(topic): update documentation for new features"

# Fix documentation issues
git commit -m "fix(docs): correct syntax errors and broken links"
```

### Configuration Files
```bash
# Configuration updates
git commit -m "chore(config): update project configuration"

# Dependency changes
git commit -m "chore(deps): update project dependencies"
```

## Repository Maintenance

### Regular Tasks
- Keep commit history clean and meaningful
- Regularly push commits to remote repository
- Monitor repository size and clean up if needed
- Update .gitignore as project evolves

### Backup and Recovery
- Remote repository serves as primary backup
- Local commits should be pushed regularly
- Use git tags for important milestones
- Document recovery procedures for critical situations

## Integration with Development Workflow

### AI Agent Integration
- AI agents must follow these conventions automatically
- Commit messages should reflect the actual changes made
- Multiple related changes can be grouped in single commits
- Each development session should result in meaningful commit history

### Human Developer Integration
- Manual commits should follow the same message format
- Review AI-generated commits for accuracy
- Can amend or squash commits if needed for clarity
- Maintain consistent workflow between AI and human contributions

This git convention ensures that all changes to the AI Hydra project are properly tracked, documented, and can be easily reviewed or reverted when necessary.