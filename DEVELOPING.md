# Developer's Guide

This guide describes how to complete various tasks you'll encounter when working
on the Bayes3D codebase.

### Commit Hooks

We use [pre-commit](https://pre-commit.com/) to manage a series of git
pre-commit hooks for the project; for example, each time you commit code, the
hooks will make sure that your python is formatted properly. If your code isn't,
the hook will format it, so when you try to commit the second time you'll get
past the hook.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks,
install `pre-commit` if you don't yet have it. I prefer using
[pipx](https://github.com/pipxproject/pipx) so that `pre-commit` stays globally
available.

```bash
pipx install pre-commit
```

Then install the hooks with this command:

```bash
pre-commit install
```

Now they'll run on every commit. If you want to run them manually, run the
following command:

```bash
pre-commit run --all-files
```

## Releasing Bayes3D

Boot up a VM... then start caliban, with the proper caliban config, TODO I'll
add this:


```bash
caliban shell
```

Clone the repo:

```bash
git clone https://github.com/probcomp/bayes3d.git && cd bayes3d
```

Install the required deps:

```bash
python -m pip install cibuildwheel==2.12.0 build==0.10.0 wheel twine
```

Build the wheel:

```bash
python -m build
```

```bash
pythonc -m twine upload \
    --repository-url https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/ \
    dist/*
```
