#!/usr/bin/env python
"""Pre-deploy check for the Modal detector. Validates, never deploys.

Answers one question: if I run `modal deploy detector/modal_app.py` right
now, will it work? It does that without creating anything, without building
an image, and without spending a cent. The only network call it ever makes
is a metadata lookup of the Modal secret, and only when a token is already
configured.

    uv run --extra deploy python scripts/modal_preflight.py
    make preflight

CI has no Modal token and should not need one, so it runs:

    uv run --extra deploy python scripts/modal_preflight.py --allow-unauthenticated

which still gates every pull request on the import, API-surface and object
graph checks while demoting the credential check to a warning.

Exit code is 0 when every required check passes, 1 otherwise. Checks marked
optional report but never fail the run. Secret *values* are never read,
printed, or logged: the script only ever learns whether a key is present.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
MODAL_APP_PATH = REPO_ROOT / "detector" / "modal_app.py"
MODAL_APP_MODULE = "detector.modal_app"

# Documented in the detector/modal_app.py docstring: the bearer token the
# LiveKit agent presents. Everything else the app reads has a default.
REQUIRED_SECRET_KEYS = ("CALLBACK_API_TOKEN",)

# Modal API surface detector/modal_app.py depends on. Checked against the
# installed version so a Modal upgrade that renames something is caught here
# instead of halfway through a deploy.
REQUIRED_MODAL_ATTRS = (
    "App",
    "Image",
    "Secret",
    "asgi_app",
)
REQUIRED_IMAGE_METHODS = ("debian_slim", "pip_install", "add_local_python_source")
REQUIRED_FUNCTION_KWARGS = (
    "image",
    "secrets",
    "min_containers",
    "max_containers",
    "timeout",
)

# Session state lives in process memory, so the app is pinned to one
# container. See the module docstring in detector/modal_app.py.
EXPECTED_CONTAINER_PINS = {"min_containers": 1, "max_containers": 1}

Status = Literal["PASS", "FAIL", "WARN", "SKIP"]

_GLYPH: dict[str, str] = {
    "PASS": "[ PASS ]",
    "FAIL": "[ FAIL ]",
    "WARN": "[ WARN ]",
    "SKIP": "[ SKIP ]",
}


@dataclass
class Check:
    """One line of the checklist."""

    title: str
    status: Status
    detail: str = ""
    hint: str = ""
    required: bool = True

    @property
    def blocking(self) -> bool:
        return self.required and self.status == "FAIL"


class Report:
    """Collects checks and prints them as a flat checklist."""

    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(
        self,
        title: str,
        status: Status,
        detail: str = "",
        hint: str = "",
        *,
        required: bool = True,
    ) -> Check:
        check = Check(
            title=title, status=status, detail=detail, hint=hint, required=required
        )
        self.checks.append(check)
        print(f"{_GLYPH[status]} {title}")
        if detail:
            for line in detail.splitlines():
                print(f"           {line}")
        return check

    @property
    def blockers(self) -> list[Check]:
        return [c for c in self.checks if c.blocking]

    def summarise(self) -> int:
        counts: dict[str, int] = dict.fromkeys(("PASS", "FAIL", "WARN", "SKIP"), 0)
        for check in self.checks:
            counts[check.status] += 1

        warned = counts["WARN"]
        print()
        print("-" * 72)
        print(
            f"{counts['PASS']} passed, {counts['FAIL']} failed, "
            f"{warned} warning{'' if warned == 1 else 's'}, "
            f"{counts['SKIP']} skipped"
        )

        blockers = self.blockers
        if not blockers:
            if warned:
                print(
                    "Required checks passed. "
                    f"{warned} non-blocking warning{'' if warned == 1 else 's'} "
                    "above, read before deploying."
                )
            else:
                print(
                    "Modal preflight OK. `modal deploy detector/modal_app.py` is clear."
                )
            return 0

        print()
        print("Blocked by:")
        for check in blockers:
            print(f"  - {check.title}")
            if check.hint:
                for line in check.hint.splitlines():
                    print(f"      {line}")
        return 1


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_python(report: Report) -> None:
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) == (3, 11):
        report.add("Python 3.11 interpreter", "PASS", f"running {version}")
    else:
        report.add(
            "Python 3.11 interpreter",
            "WARN",
            f"running {version}, the Modal image is built on 3.11",
            hint="Run through uv so the pinned interpreter is used: "
            "uv run python scripts/modal_preflight.py",
            required=False,
        )


def check_modal_import(report: Report) -> ModuleType | None:
    try:
        import modal
    except ImportError as exc:
        report.add(
            "modal package importable",
            "FAIL",
            str(exc),
            hint="The deploy extra is not installed. Run:\n"
            "  uv sync --extra deploy\n"
            "then re-run this script with `uv run`.",
        )
        return None
    except Exception as exc:  # defensive: a broken install, not a missing one
        report.add(
            "modal package importable",
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            hint="The modal package is installed but failed to import. "
            "Try: uv sync --reinstall --extra deploy",
        )
        return None

    version = getattr(modal, "__version__", "unknown")
    major = version.split(".", 1)[0]
    if major.isdigit() and int(major) < 1:
        report.add(
            "modal package importable",
            "FAIL",
            f"modal {version}",
            hint="pyproject.toml declares modal>=1.0. Upgrade with: "
            "uv sync --extra deploy --upgrade-package modal",
        )
        return modal

    report.add("modal package importable", "PASS", f"modal {version}")
    return modal


def _auth_from_env() -> list[str]:
    return [
        name
        for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
        if os.environ.get(name, "").strip()
    ]


def _auth_from_config(modal: ModuleType) -> tuple[bool, str]:
    """Look for a token in the Modal config file. Returns (found, detail)."""
    try:
        from modal import config as modal_config
    except Exception as exc:
        return False, f"could not read Modal config: {type(exc).__name__}: {exc}"

    config_path = Path(str(getattr(modal_config, "user_config_path", "")))
    if not config_path or not config_path.exists():
        return False, f"no config file at {config_path or '~/.modal.toml'}"

    try:
        token_id = modal_config.config.get("token_id")
        token_secret = modal_config.config.get("token_secret")
    except Exception as exc:
        return False, f"{config_path} unreadable: {type(exc).__name__}: {exc}"

    profile = os.environ.get("MODAL_PROFILE", "") or "active profile"
    if token_id and token_secret:
        return True, f"token found in {config_path} ({profile})"
    return False, f"{config_path} exists but has no token_id/token_secret ({profile})"


def check_modal_auth(
    report: Report, modal: ModuleType, *, required: bool = True
) -> bool:
    """True when a credential is present. Never reports the credential."""
    env_names = _auth_from_env()
    if len(env_names) == 2:
        report.add(
            "Modal authenticated",
            "PASS",
            "token supplied by MODAL_TOKEN_ID and MODAL_TOKEN_SECRET (values not read)",
        )
        return True

    found, detail = _auth_from_config(modal)
    if found:
        report.add("Modal authenticated", "PASS", detail)
        return True

    partial = (
        f"\n{env_names[0]} is set but its pair is not" if len(env_names) == 1 else ""
    )
    report.add(
        "Modal authenticated",
        "FAIL" if required else "WARN",
        detail + partial,
        hint="Authenticate this machine once:\n"
        "  uv run --extra deploy modal token new\n"
        "or export MODAL_TOKEN_ID and MODAL_TOKEN_SECRET for CI.",
        required=required,
    )
    return False


def check_modal_api_surface(report: Report, modal: ModuleType) -> None:
    import inspect

    missing: list[str] = []

    for attr in REQUIRED_MODAL_ATTRS:
        if not hasattr(modal, attr):
            missing.append(f"modal.{attr}")

    if hasattr(modal, "Image"):
        for method in REQUIRED_IMAGE_METHODS:
            if not hasattr(modal.Image, method):
                missing.append(f"modal.Image.{method}")

    if hasattr(modal, "Secret") and not hasattr(modal.Secret, "from_name"):
        missing.append("modal.Secret.from_name")

    if hasattr(modal, "App"):
        try:
            params = inspect.signature(modal.App.function).parameters
        except (TypeError, ValueError) as exc:
            missing.append(f"modal.App.function signature unreadable ({exc})")
        else:
            missing.extend(
                f"modal.App.function(..., {kwarg}=)"
                for kwarg in REQUIRED_FUNCTION_KWARGS
                if kwarg not in params
            )

    if missing:
        report.add(
            "Modal API symbols used by modal_app.py",
            "FAIL",
            "missing from the installed version:\n"
            + "\n".join(f"  {name}" for name in missing),
            hint="detector/modal_app.py targets a different Modal version. "
            "Either pin modal in pyproject.toml or update modal_app.py.",
        )
        return

    checked = (
        len(REQUIRED_MODAL_ATTRS)
        + len(REQUIRED_IMAGE_METHODS)
        + 1
        + len(REQUIRED_FUNCTION_KWARGS)
    )
    report.add(
        "Modal API symbols used by modal_app.py",
        "PASS",
        f"{checked} symbols and keyword arguments present",
    )


def check_modal_app_builds(report: Report, modal: ModuleType) -> object | None:
    """Import the deployment module and inspect the object graph it builds."""
    if not MODAL_APP_PATH.exists():
        report.add(
            "detector/modal_app.py builds its object graph",
            "FAIL",
            f"no such file: {MODAL_APP_PATH}",
            hint="Expected the Modal entrypoint at detector/modal_app.py.",
        )
        return None

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    try:
        import importlib

        module = importlib.import_module(MODAL_APP_MODULE)
    except Exception as exc:
        report.add(
            "detector/modal_app.py builds its object graph",
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            hint="The deployment module does not import. Reproduce with:\n"
            f"  uv run python -c 'import {MODAL_APP_MODULE}'",
        )
        return None

    problems: list[str] = []
    notes: list[str] = []

    app = getattr(module, "app", None)
    if app is None:
        problems.append("module defines no `app`")
    elif not isinstance(app, modal.App):
        problems.append(f"`app` is {type(app).__name__}, expected modal.App")
    else:
        notes.append(f"modal.App name: {app.name!r}")

    image = getattr(module, "image", None)
    if image is None:
        problems.append("module defines no `image`")
    elif not isinstance(image, modal.Image):
        problems.append(f"`image` is {type(image).__name__}, expected modal.Image")
    else:
        notes.append("modal.Image built (not yet pushed, no build triggered)")

    # Deliberately does not touch app.registered_functions: that attribute is
    # deprecated and goes away in modal 1.7. The entrypoint object itself is
    # the thing `modal deploy` needs.
    entrypoint = getattr(module, "fastapi_app", None)
    if entrypoint is None:
        problems.append("module defines no `fastapi_app` entrypoint")
    else:
        notes.append(f"asgi entrypoint: fastapi_app ({type(entrypoint).__name__})")

    if problems:
        report.add(
            "detector/modal_app.py builds its object graph",
            "FAIL",
            "\n".join(problems),
            hint="Fix detector/modal_app.py so it exposes `app`, `image`, "
            "and the `fastapi_app` asgi entrypoint.",
        )
        return None

    report.add(
        "detector/modal_app.py builds its object graph", "PASS", "\n".join(notes)
    )
    return module


# --------------------------------------------------------------------------
# static read of modal_app.py, so the secret name is never hardcoded twice
# --------------------------------------------------------------------------


def _parse_modal_app() -> tuple[ast.Module | None, str]:
    try:
        source = MODAL_APP_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"cannot read {MODAL_APP_PATH}: {exc}"
    try:
        return ast.parse(source, filename=str(MODAL_APP_PATH)), ""
    except SyntaxError as exc:
        return None, f"{MODAL_APP_PATH} has a syntax error on line {exc.lineno}"


def _secret_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "from_name"):
            continue
        owner = func.value
        owner_name = owner.attr if isinstance(owner, ast.Attribute) else None
        if isinstance(owner, ast.Name):
            owner_name = owner.id
        if owner_name != "Secret":
            continue
        for arg in node.args[:1]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                names.append(arg.value)
    return sorted(set(names))


def _container_pins(tree: ast.Module) -> dict[str, object]:
    pins: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "function"):
            continue
        for kw in node.keywords:
            if kw.arg in EXPECTED_CONTAINER_PINS and isinstance(kw.value, ast.Constant):
                pins[kw.arg] = kw.value.value
    return pins


def check_container_pin(report: Report, tree: ast.Module | None) -> None:
    if tree is None:
        report.add(
            "single-container pin still in place",
            "SKIP",
            "could not parse detector/modal_app.py",
            required=False,
        )
        return

    pins = _container_pins(tree)
    drift = [
        f"{key}={pins.get(key, 'unset')!r}, expected {expected!r}"
        for key, expected in EXPECTED_CONTAINER_PINS.items()
        if pins.get(key) != expected
    ]
    if drift:
        report.add(
            "single-container pin still in place",
            "WARN",
            "\n".join(drift),
            hint="Session state lives in process memory. Scaling past one "
            "container drops sessions until SessionStore is backed by Redis "
            "or a Modal Dict.",
            required=False,
        )
        return

    report.add(
        "single-container pin still in place",
        "PASS",
        "min_containers=1, max_containers=1 (session state is in-process)",
        required=False,
    )


def check_secret(
    report: Report,
    modal: ModuleType,
    *,
    skip_reason: str,
    secret_names: list[str],
    parse_error: str,
) -> None:
    if parse_error:
        report.add(
            "Modal secret exists with required keys",
            "SKIP",
            parse_error,
            required=False,
        )
        return

    if not secret_names:
        report.add(
            "Modal secret exists with required keys",
            "WARN",
            "detector/modal_app.py references no modal.Secret.from_name(...)",
            hint="If the app no longer needs a secret, drop "
            "REQUIRED_SECRET_KEYS from this script.",
            required=False,
        )
        return

    wanted = ", ".join(REQUIRED_SECRET_KEYS)
    for name in secret_names:
        title = f"Modal secret {name!r} exists and contains {wanted}"

        if skip_reason:
            report.add(title, "SKIP", skip_reason, required=False)
            continue

        try:
            secret = modal.Secret.from_name(
                name, required_keys=list(REQUIRED_SECRET_KEYS)
            )
            secret.hydrate()
        except Exception as exc:
            kind = type(exc).__name__
            message = str(exc).strip() or kind
            if kind == "NotFoundError":
                hint = (
                    f"Create it (the value is yours, not printed here):\n"
                    f"  modal secret create {name} CALLBACK_API_TOKEN=<long-random-string>"
                )
            elif kind == "AuthError":
                hint = (
                    "The token is present but rejected. Re-authenticate:\n"
                    "  uv run --extra deploy modal token new"
                )
            else:
                hint = (
                    f"Modal rejected the lookup ({kind}). Confirm the secret "
                    f"and its keys with: modal secret list"
                )
            report.add(title, "FAIL", f"{kind}: {message}", hint=hint)
            continue

        report.add(
            title,
            "PASS",
            "secret resolved and required keys present (values not read)",
        )


def check_deploy_extra_declared(report: Report) -> None:
    """modal must be a declared dependency, not something a dev happens to have."""
    pyproject = REPO_ROOT / "pyproject.toml"
    try:
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add(
            "modal declared in the deploy extra",
            "SKIP",
            f"cannot read pyproject.toml: {type(exc).__name__}: {exc}",
            required=False,
        )
        return

    extras = data.get("project", {}).get("optional-dependencies", {})
    deploy = extras.get("deploy", [])
    if any(str(spec).replace(" ", "").startswith("modal") for spec in deploy):
        report.add(
            "modal declared in the deploy extra",
            "PASS",
            f"[project.optional-dependencies] deploy = {list(deploy)}",
            required=False,
        )
        return

    report.add(
        "modal declared in the deploy extra",
        "WARN",
        "pyproject.toml does not list modal under the `deploy` extra",
        hint='Add it so deploys are reproducible: deploy = ["modal>=1.0"]',
        required=False,
    )


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate everything a Modal deploy of the detector needs. "
        "Deploys nothing, builds nothing, spends nothing."
    )
    parser.add_argument(
        "--skip-remote",
        action="store_true",
        help="Do not contact Modal, even when a token is configured.",
    )
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="Report missing Modal credentials as a warning instead of a "
        "failure, and skip the secret lookup. For CI, which has no Modal "
        "token but still wants the import, API-surface and object-graph "
        "checks to gate every pull request.",
    )
    args = parser.parse_args(argv)

    print("Modal deploy preflight")
    print(f"repo: {REPO_ROOT}")
    print("-" * 72)

    report = Report()
    check_python(report)
    check_deploy_extra_declared(report)

    modal = check_modal_import(report)
    if modal is None:
        print()
        print("Stopping early: the rest of the checks all need the modal package.")
        return report.summarise()

    check_modal_api_surface(report, modal)
    check_modal_app_builds(report, modal)

    tree, parse_error = _parse_modal_app()
    check_container_pin(report, tree)

    authenticated = check_modal_auth(
        report, modal, required=not args.allow_unauthenticated
    )
    if not authenticated:
        skip_reason = "cannot check, not authenticated"
    elif args.skip_remote:
        skip_reason = "cannot check, --skip-remote was given"
    else:
        skip_reason = ""

    check_secret(
        report,
        modal,
        skip_reason=skip_reason,
        secret_names=_secret_names(tree) if tree is not None else [],
        parse_error=parse_error,
    )

    return report.summarise()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130) from None
