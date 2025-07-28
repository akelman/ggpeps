import os
import nox

# Define the minimal nox version required to run
nox.needs_version = ">= 2024.3.2"


@nox.session
def lint(session):
    session.install("flake8")
    session.run(
        "flake8",
        "--exclude",
        ".nox,*.egg,build,data",
        "--select",
        "E,W,F",
        ".",
        "--extend-ignore",
        "E203, W503",  # whitespace in slices, line break before binary operator
        "--max-line-length",
        "119",
    )


@nox.session
def typing(session):
    """Perform static type checking."""

    session.install(".")
    session.install("mypy")
    session.run(
        "mypy",
        "--install-types",  # install missing types for third-party packages
        "--non-interactive",  # don't ask user for confirmation before installing missing types
        "src/ggpeps/caching.py",
        "src/ggpeps/evaluator_manager.py",
        "src/ggpeps/evaluator.py",
        "src/ggpeps/exacteval.py",
        "src/ggpeps/lattice.py",
        "src/ggpeps/gauge.py",
    )
    # TODO: Add passing files (eventually should be entire repo)


@nox.session
def build_and_check_dists(session):
    session.install("build", "check-manifest >= 0.42", "twine")

    # TODO: Enable again eventually in case we need a manifest file
    # session.run("check-manifest", "--ignore", "noxfile.py,tests/**")
    session.run("python", "-m", "build")
    # session.run("python", "-m", "twine", "check", "dist/*")


@nox.session(python=["3"])
def tests_numpy(session):

    session.install("-e", ".")

    session.run("python", "-m", "unittest", env={"GGPEPS_BACKEND": "numpy"})


@nox.session(python=["3"])
def tests_jax(session):
    session.install("-e", ".")

    session.run("python", "-m", "unittest", env={"GGPEPS_BACKEND": "jax"})


@nox.session
def coverage(session):
    session.install("coverage")
    session.run("coverage")
