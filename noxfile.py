"""This module implements our CI function calls."""

import nox


nox.options.sessions = ["lint", "typing", "format"]


@nox.session(name="test")
def run_test(session):
    """Run pytest.

    Args:
        session: test
    """
    session.install(".")
    session.install("pytest")
    session.install("torchvision")
    session.run("pytest")


@nox.session(name="limited-test", python='3.8')
def run_small_test(session):
    """Run ioucalc test.

    Args:
        session: test with full dependencies
    """
    session.install(".")
    session.install("pytest")
    session.install("torchvision")
    session.install("pandas")
    session.install("bs4")
    session.install("torch")
    session.install("pillow")
    session.install("tqdm")
    session.install("scikit-learn")
    session.install("lxml")
    session.install("lightning")
    session.install("transformers")
    session.install("tikzplotlib")
    if session.posargs:
        test_files = session.posargs
    else:
        test_files = ["tests/split_test.py", "tests/ioucalc_test.py"]
    session.run("pytest", *test_files)


@nox.session(name="fast-test")
def run_test_fast(session):
    """Run pytest.

    Args:
        session: test
    """
    session.install(".")
    session.install("pytest")
    session.install("torchvision")
    session.run("pytest", "-m", "not slow")


@nox.session(name="lint")
def lint(session):
    """Check code conventions.

    Args:
        session: lint
    """
    session.install("flake8")
    session.install(
        "flake8-black",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-broken-line",
        "pep8-naming",
        "pydocstyle",
        "darglint",
    )
    session.run("flake8", "src/historicdocumentprocessing", "tests", "noxfile.py")


@nox.session(name="typing")
def mypy(session):
    """Check type hints.

    Args:
        session: typing
    """
    session.install(".")
    session.install("mypy")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        "--no-warn-return-any",
        "--explicit-package-bases",
        "src",
    )


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically.

    Args:
        session: formatting
    """
    session.install("black")
    session.install("isort")
    session.run("isort", "src", "tests", "noxfile.py")
    session.run("black", "src", "tests", "noxfile.py")


@nox.session(name="coverage")
def check_coverage(session):
    """Check test coverage and generate a html report.

    Args:
        session: coverage
    """
    session.install(".")
    session.install("pytest")
    session.install("coverage")
    try:
        session.run("coverage", "run", "-m", "pytest")
    finally:
        session.run("coverage", "html")


@nox.session(name="coverage-clean")
def clean_coverage(session):
    """Remove the code coverage website.

    Args:
        session: lint
    """
    session.run("rm", "-r", "htmlcov", external=True)


@nox.session(name="build")
def build(session):
    """Build a pip package.

    Args:
        session: build pip package
    """
    session.install("wheel")
    session.install("setuptools")
    session.run("python", "setup.py", "-q", "sdist", "bdist_wheel")


@nox.session(name="finish")
def finish(session):
    """Finish this version increase the version number and upload to pypi.

    Args:
        session: finish and upload
    """
    session.install("bump2version")
    session.install("twine")
    session.run("bumpversion", "release", external=True)
    build(session)
    session.run("twine", "upload", "--skip-existing", "dist/*", external=True)
    session.run("git", "push", external=True)
    session.run("bumpversion", "patch", external=True)
    session.run("git", "push", external=True)


@nox.session(name="check-package")
def pyroma(session):
    """Run pyroma to check if the package is ok.

    Args:
        session: check package
    """
    session.install("pyroma")
    session.run("pyroma", ".")
