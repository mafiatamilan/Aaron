# 0004 — Canonicalise paths, do not only inspect them lexically

**Status:** accepted

## Context

`read_file` and `search_files` must not escape `security.allowed_roots`.

## Decision

Resolve the path with `canonicalize` (following symlinks), then check that the
result is inside a canonicalised root. Search never follows symlinks at all.

## Reasoning

A lexical check for `..` passes this attack:

```
root/link -> /etc
read_file("link/passwd")
```

There is no `..` in the string, and no lexical check will see that `link` is a
symlink pointing outside. Canonicalisation catches it because the kernel
resolves the link before the containment test runs.

Not following symlinks during search additionally removes both the escape case
and the possibility of a directory loop.

## Consequences

* A path under a root that becomes unreadable mid-call is rejected rather than
  guessed at.
* Legitimate symlink usage inside a root is refused. That is the intended
  trade: a symlink is a redirect, and a redirect is the thing being defended
  against.
