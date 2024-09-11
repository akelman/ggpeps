import os
import nox

# Define the minimal nox version required to run
nox.options.needs_version = ">= 2024.3.2"


@nox.session
def lint(session):
    session.install("flake8")
    session.run(
        "flake8", "--exclude", ".nox,*.egg,build,data", "--select", "E,W,F", "."
    )


@nox.session
def build_and_check_dists(session):
    session.install("build", "check-manifest >= 0.42", "twine")

    # TODO: Enable again eventually in case we need a manifest file
    # session.run("check-manifest", "--ignore", "noxfile.py,tests/**")
    session.run("python", "-m", "build")
    # session.run("python", "-m", "twine", "check", "dist/*")


@nox.session(python=["3.8"])
def tests_jax(session):
    build_and_check_dists(session)

    generated_files = os.listdir("dist/")
    generated_sdist = os.path.join("dist/", generated_files[1])

    session.install(generated_sdist)

    session.run("python", "-m", "unittest", env={"GGPEPS_BACKEND": "jax"})


@nox.session(python=["3.8"])
def tests_numpy(session):
    build_and_check_dists(session)

    generated_files = os.listdir("dist/")
    generated_sdist = os.path.join("dist/", generated_files[1])

    session.install(generated_sdist)

    session.run("python", "-m", "unittest", env={"GGPEPS_BACKEND": "numpy"})


@nox.session(python=["3.8"])
def tests(session):
    build_and_check_dists(session)

    generated_files = os.listdir("dist/")
    generated_sdist = os.path.join("dist/", generated_files[1])

    session.install(generated_sdist)

    session.run("python", "-m", "unittest", env={"GGPEPS_BACKEND": "numpy"})

    session.run("python", "-m", "unittest", env={"GGPEPS_BACKEND": "jax"})
    session.notify("coverage")


@nox.session
def coverage(session):
    session.install("coverage")
    session.run("coverage")
