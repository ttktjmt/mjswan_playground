# Agents

- Every pull request ends with a `/simplify-comments` pass over its whole diff,
  `git diff origin/main...HEAD`: every file it changes, not only the latest commit. Run it
  once the pull request's last change is made, push what it changes, and run it again
  after any later change. Without the command, follow
  `.claude/commands/simplify-comments.md`.
