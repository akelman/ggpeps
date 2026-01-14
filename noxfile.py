import nox

# Define the minimal nox version required to run
nox.needs_version = ">= 2024.3.2"


@nox.session
def lint(session):
    """Check code style"""
    session.install("flake8")
    session.run(
        "flake8",
        "--exclude",
        ".nox,*.egg,build,data,.*",
        "--select",
        "E,W,F",
        ".",
        "--extend-ignore",
        # whitespace in slices, line break before binary operator, multiple leading ##, imports not at top of file
        "E203, W503, E266, E402",
        "--max-line-length",
        "120",
    )


@nox.session
def typing(session):
    """Perform static type checking."""

    session.install(".")
    session.install("mypy")

    session.run(
        "mypy",
        "--enable-incomplete-feature=PreciseTupleTypes",
        "--install-types",  # install missing types for third-party packages
        "--non-interactive",  # don't ask user for confirmation before installing missing types
        "src/ggpeps/",
    )


@nox.session
def build_and_check_dists(session):
    session.install("build", "check-manifest >= 0.42", "twine")

    # TODO: Enable again eventually in case we need a manifest file
    # session.run("check-manifest", "--ignore", "noxfile.py,tests/**")
    session.run("python", "-m", "build")
    # session.run("python", "-m", "twine", "check", "dist/*")


@nox.session(python=["3"])
def tests_numpy(session):
    """Run the unit tests with numpy backend."""

    session.install("-e", ".")
    session.run("python", "-m", "unittest", env={"GGPEPS_BACKEND": "numpy"})


@nox.session(python=["3"])
def tests_jax(session):
    """Run the unit tests with jax backend."""
    session.install("-e", ".")
    session.run("python", "-m", "unittest", env={"GGPEPS_BACKEND": "jax"})


@nox.session(python=["3"])
def jax_eval(session):
    """Run a simple eval with jax, to make sure jax with jit is working fine."""

    session.install("-e", ".")

    session.run(
        "python",
        "manager.py",
        "eval-mc",
        "Z2",
        "--L",
        "2",
        "--g",
        "1.0",
        "--int",
        "1.0",
        "--mass",
        "1.0",
        "--chem",
        "1.0",
        "--el_links",
        "0",
        "1",
        "--relax_u1",
        "--unitcell_size",
        "2",
        "--ncopy",
        "2",
        "--num_pg_layer",
        "1",
        "--num_fermionic_layer",
        "1",
        "--warmup_steps",
        "10",
        "--meas_steps",
        "10",
        "--compute_grads",
        env={"GGPEPS_BACKEND": "jax"},
    )


@nox.session
def coverage(session):
    session.install("coverage")
    session.run("coverage")
