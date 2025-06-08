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
        "E203",  # whitespace around : in slices
        "--max-line-length",
        "89",
    )


@nox.session
def build_and_check_dists(session):
    session.install("build", "check-manifest >= 0.42", "twine")

    # TODO: Enable again eventually in case we need a manifest file
    # session.run("check-manifest", "--ignore", "noxfile.py,tests/**")
    session.run("python", "-m", "build")
    # session.run("python", "-m", "twine", "check", "dist/*")


@nox.session(python=["3"])
def tests_jax(session):
    session.install("-e", ".")

    session.run("python", "-m", "unittest", env={"GGPEPS_BACKEND": "jax"})


@nox.session(python=["3"])
def tests_numpy(session):

    session.install("-e", ".")

    session.run("python", "-m", "unittest", env={"GGPEPS_BACKEND": "numpy"})


@nox.session
def coverage(session):
    session.install("coverage")
    session.run("coverage")
