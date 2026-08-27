# Contributor guidelines
When contributing to this repository, please first [create an issue](https://github.com/scientificcomputing/dolfinx-adjoint/issues/new/choose) containing information about the missing feature or the bug that you would like to fix. Here you can discuss the change you want to make with the maintainers of the repository.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## New contributor guide

To get an overview of the project, read the [documentation](https://scientificcomputing.github.io/dolfinx-adjoint). Here are some resources to help you get started with open source contributions:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

## Pull Request Process


### Pull Request

- When you're finished with the changes, create a pull request, also known as a PR. It is also OK to create a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/) from the very beginning. Once you are done you can click on the ["Ready for review"] button. You can also [request a review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review) from one of the maintainers.
- Don't forget to [link PR to the issue that you opened ](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).
- Enable the checkbox to [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a merge.
Once you submit your PR, a team member will review your proposal. We may ask questions or request for additional information.
- We may ask for changes to be made before a PR can be merged, either using [suggested changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request) or pull request comments. You can apply suggested changes directly through the UI. You can make any other changes in your fork, then commit them to your branch.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this [git tutorial](https://lab.github.com/githubtraining/managing-merge-conflicts) to help you resolve merge conflicts and other issues.
- Please make sure that all tests are passing, github pages renders nicely, and code coverage are are not lower than before your contribution. You see the different github action workflows by clicking the "Action" tab in the GitHub repository.

Note that for a pull request to be accepted, it has to pass all the tests on CI, which includes:
- `mypy`: typechecking
- `ruff`: Code formatting
- `pytest`: Successfull execution of all tests in the `tests` folder.


## Design notes, and working with coding agents

The reasoning behind the design — a domain glossary, architecture decision records, feature
specs, the papers the algorithms follow, and memory for coding agents — is kept in a separate,
internal repository,
[dolfinx-adjoint-knowledge](https://github.com/scientificcomputing/dolfinx-adjoint-knowledge),
so that this one stays free of files most contributors do not need.

**None of it is required.** You can build, test and contribute without it, and nothing in the
package imports or reads it.

### Linking it in

If you have access, clone it *beside* this repository and link it in from the repository root.
The paths it provides are the ones this project's tooling and agent instructions already look
for, so linking them makes everything resolve in place:

```bash
git clone git@github.com:scientificcomputing/dolfinx-adjoint-knowledge.git ../dolfinx-adjoint-knowledge

ln -s ../dolfinx-adjoint-knowledge/CONTEXT.md   CONTEXT.md
ln -s ../dolfinx-adjoint-knowledge/CLAUDE.md    CLAUDE.md
ln -s ../dolfinx-adjoint-knowledge/scratch      .scratch
ln -s ../dolfinx-adjoint-knowledge/references   references
ln -s ../../dolfinx-adjoint-knowledge/adr       docs/adr
ln -s ../../dolfinx-adjoint-knowledge/agents    docs/agents

# Keep them out of this repository's history. `.git/info/exclude` is per-clone and is not
# itself versioned, so this has to be done again in every clone. Do not give the patterns a
# trailing slash: `.scratch/` matches a directory and stops matching once it is a symlink.
printf '%s\n' CONTEXT.md CLAUDE.md .scratch references docs/adr docs/agents >> .git/info/exclude
```

The development container mounts only this repository, so these links point outside the mount
and are not readable from inside it. That is harmless — building and testing touch only `src/`,
`tests/` and `demos/`.

### What an agent picks up once it is linked

[Claude Code](https://claude.com/claude-code) reads most of this on its own:

| Path          | What it gives the agent                                                    |
| ------------- | -------------------------------------------------------------------------- |
| `CLAUDE.md`   | Project instructions, read automatically at the start of every session.     |
| `CONTEXT.md`  | The domain glossary: the words this project uses, and the ones it avoids.   |
| `docs/adr/`   | Why things are the way they are, so decisions are not silently re-litigated.|
| `docs/agents/`| Where issues live, and the triage vocabulary the skills expect.             |
| `.scratch/`   | Specs and issues, one directory per feature.                                |
| `references/` | The papers to follow rather than improvising.                              |

Agent memory is stored per project outside the repository, and is symlinked to the knowledge
repository so that it is versioned rather than left in a cache directory:

```bash
# The directory name is the absolute path to your checkout with "/" replaced by "-"
ln -s "$PWD/../dolfinx-adjoint-knowledge/memory" \
      ~/.claude/projects/"$(pwd | tr / -)"/memory
```

### Skills

The repository-scoped skills the agent instructions refer to — `grilling`, `domain-modeling`,
`code-review`, `to-spec` and others — are vendored under `.agents/skills/`, exposed to Claude
Code through `.claude/skills/`, and pinned in `skills-lock.json`. They come from
[mattpocock/skills](https://github.com/mattpocock/skills) and are not part of the knowledge
repository, since they are upstream content rather than anything specific to this project.

`setup-matt-pocock-skills` scaffolds the per-repository configuration those skills assume — the
issue tracker layout, the triage vocabulary and the domain doc paths, which is what
`docs/agents/` contains. It configures the repository; it does not install the skills.


### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of experience,
nationality, personal appearance, race, religion, or sexual identity and
orientation.
