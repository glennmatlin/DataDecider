# AI Agent Guidelines

This document defines the operational guidelines, tool usage standards, and behavioral rules for AI agents working in shell-based and code development workflows.

---

## Core Principles

1. **Professional Anonymity**: Never mention AI, Claude, or any AI assistant names in:
   - Git commit messages
   - Code comments (unless explaining technical logic)
   - Documentation
   - Any user-facing output

2. **Objective Communication**:
   - Avoid hyperbole, over-complementary language, or "encouraging" tone
   - State facts and progress objectively
   - Do not overstate completed work or project status

---

## CLI Tool Standards

### File Operations

**Search & Discovery**
- **Content search**: Use `rg` (ripgrep) exclusively
  ```bash
  rg "pattern" path/to/dir
  rg -i "case_insensitive" .
  rg -t py "import pandas" src/
  ```

- **File/directory search**: Use `fd` for fast filesystem queries
  ```bash
  fd "\.py$" src/
  fd -t d "test" .
  ```

**File Viewing**
- **Interactive viewing**: Use `bat` for syntax-highlighted display
  ```bash
  bat config.yaml
  bat -n script.py  # with line numbers
  ```

- **Piped output**: Use `cat` when feeding to other tools
  ```bash
  cat data.json | jq '.results[]'
  cat script.sh | shellcheck -
  ```

### Directory & Version Control

- **Directory listings**: Use `eza` instead of `ls`
  ```bash
  eza -lah --git
  eza --tree --level=2
  ```

- **Git diffs**: Use `delta` as the diff pager
  ```bash
  git diff --color | delta
  git show HEAD~1 | delta
  ```

### Code Quality & Validation

**Shell Scripts**
1. **Always lint** before execution:
   ```bash
   shellcheck script.sh
   ```

2. **Always format** for consistency:
   ```bash
   shfmt -w script.sh
   ```

**Python Code**
- **Always use** `ruff` for linting and formatting:
  ```bash
  ruff check .
  ruff format .
  ```

### Data Processing

- **JSON**: Use `jq` for all JSON operations
  ```bash
  jq '.items[] | select(.active == true)' data.json
  jq -r '.users[].email' users.json
  ```

- **YAML**: Use `yq` for YAML manipulation
  ```bash
  yq eval '.services.*.port' docker-compose.yml
  yq eval '.version = "3.9"' -i config.yaml
  ```

---

## Python Development Workflow

### Package Management

**Critical Rule**: Always use `uv` package manager. Never run Python directly.

```bash
# Correct usage
uv pip install pandas
uv run python script.py

# NEVER do this
python script.py
pip install package
```

---

## Git Commit Standards

1. **No signatures or attribution**
   - No names, handles, or "Created by" messages
   - No AI/Claude references

2. **Commit message format**:
   ```
   type: brief objective description

   - Specific change 1
   - Specific change 2
   ```

3. **Example good commits**:
   ```
   fix: correct data validation logic in parser

   - Add null check for optional fields
   - Update regex pattern for email validation
   ```

---

## Workflow Best Practices

### Before Any Operation

1. **Verify environment**: Check tool availability
2. **Validate syntax**: Use appropriate linters
3. **Test incrementally**: Run small verification steps

### File Manipulation

1. **Always backup** before major changes:
   ```bash
   cp important.conf important.conf.bak
   ```

2. **Use version control** for tracking:
   ```bash
   git add -p  # Review changes before staging
   ```

### Error Handling

- Anticipate common failure modes
- Provide clear error messages
- Suggest actionable fixes

---

## Integration Guidelines

### CI/CD Workflows

Embed quality checks in all pipelines:
```yaml
steps:
  - shellcheck scripts/*.sh
  - shfmt -d scripts/*.sh
  - ruff check .
  - ruff format --check .
```

### Documentation

- Keep technical documentation factual
- Avoid subjective quality assessments
- Focus on functionality and usage

---

## LOGBOOK.md Protocol

You must maintain a `LOGBOOK.md` file throughout your work. This is a permanent, append-only record - think of it as an immutable audit trail.

**Critical Rules:**
1. **NEVER delete or modify existing entries** - The logbook is append-only
2. **Always add new entries at the END of the file**
3. **Include timestamps for each entry**
4. **Document EVERYTHING** - mistakes, dead ends, solutions, realizations, todos

**Entry Format:**

```
[YYYY-MM-DD HH:MM] Entry Title

Context: What you were trying to do

Action: What you actually did

Result: What happened (including errors/failures)

Learning: Any insights or next steps
```

**What to Log:**
- Initial understanding of the task
- Each approach attempted (successful or not)
- Errors encountered and their full messages
- Solutions discovered
- Workarounds implemented
- TODOs and future considerations
- Assumptions made
- External resources consulted
- Any "aha!" moments or pattern recognitions
- Dead ends (these are valuable!)

**Remember:** The logbook is a historical record. Even if you later discover a logged assumption was wrong or an approach was flawed, DO NOT go back and edit. Instead, add a new entry explaining the correction. The journey is as important as the destination.

Create LOGBOOK.md if it doesn't exist, and update it frequently as you work.

---

## Summary

When completing any task:
- [ ] Use modern CLI tools (rg, fd, bat, eza)
- [ ] Lint all code (shellcheck, ruff)
- [ ] Format consistently (shfmt, ruff format)
- [ ] Use uv for all Python operations
- [ ] Git commits contain no AI/name references
- [ ] Communication is objective and factual
- [ ] Add your updates to LOGBOOK.md without modifying old entries

@insert /docs/