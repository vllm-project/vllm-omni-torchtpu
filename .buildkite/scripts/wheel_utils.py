#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions for building and publishing the release wheel.

Adapted from vllm-project/vllm-torchtpu .buildkite/scripts/wheel_utils.py.
build_wheel.sh uses the build checks; publish_wheel.sh uses the publish checks
and the PyPI Trusted Publishing (Google Cloud OIDC) token exchange for the one
pure-Python vllm-omni-torchtpu wheel and its sdist.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile

from packaging.utils import (
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import Version

# The only top-level package the wheel may ship (besides its .dist-info).
PACKAGE = "vllm_omni_torchtpu"


def verify_built_wheel(wheel_path: str, expected_version: str) -> None:
    """Verify the built wheel's filename version and its py3-none-any tag."""
    name, version, _, tags = parse_wheel_filename(os.path.basename(wheel_path))
    if version != Version(expected_version):
        sys.exit(
            f"ERROR: wheel version {version} != release version {expected_version}"
        )
    if {str(t) for t in tags} != {"py3-none-any"}:
        sys.exit(
            f"ERROR: wheel is tagged {sorted(map(str, tags))}, not py3-none-any; "
            "vllm-omni-torchtpu is pure Python."
        )
    print(f"Wheel: {name} {version}")


def verify_wheel_contents(wheel_path: str) -> None:
    """Verify the wheel holds only the package and its .dist-info.

    Catches setuptools picking up tests/, docs/ or .buildkite/ by mistake.
    """
    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()
    stray = [
        n
        for n in names
        if not n.startswith(f"{PACKAGE}/")
        and not (n.split("/", 1)[0].endswith(".dist-info"))
    ]
    if stray:
        sys.exit(f"ERROR: {os.path.basename(wheel_path)} ships stray files: {stray}")
    if not any(n.startswith(f"{PACKAGE}/") and n.endswith(".py") for n in names):
        sys.exit(f"ERROR: {os.path.basename(wheel_path)} has no {PACKAGE} modules.")
    print(f"Wheel contents: {len(names)} entries, all under {PACKAGE}/ or .dist-info")


def verify_publish_dist(dist_path: str, project: str, expected_version: str) -> None:
    """Verify a wheel or sdist matches the target
    PyPI project name and release version."""
    filename = os.path.basename(dist_path)
    if filename.endswith(".whl"):
        name, version, _, _ = parse_wheel_filename(filename)
    else:
        name, version = parse_sdist_filename(filename)
    if name != canonicalize_name(project):
        sys.exit(f"ERROR: {filename} is '{name}' but PUBLISH_PYPI is '{project}'.")
    if version != Version(expected_version):
        sys.exit(
            f"ERROR: {filename} version {version} "
            f"!= release version {expected_version}."
        )
    print(f"Publishing {filename}")


def fetch_oidc_audience(index_base: str) -> None:
    """Query the PyPI OIDC audience endpoint and print the audience string."""
    with urllib.request.urlopen(f"{index_base}/_/oidc/audience", timeout=30) as r:
        audience = json.load(r).get("audience")
    if not audience:
        sys.exit("ERROR: Failed to parse audience from OIDC audience endpoint.")
    print(audience)


def decode_jwt_claims(token: str) -> dict:
    """The (unverified) claims of a JWT; PyPI verifies the signature."""
    parts = token.strip().split(".")
    if len(parts) != 3:
        sys.exit("ERROR: identity token is not a JWT.")
    payload = parts[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def check_gcp_oidc_claims(claims: dict, expected_aud: str) -> str | None:
    """None if the claims are a Google identity for `expected_aud`, else why not."""
    if claims.get("iss") != "https://accounts.google.com":
        return (
            f"identity token is from {claims.get('iss')!r}, not Google Cloud. "
            "Run this step on a GCE VM agent with a service account."
        )
    if claims.get("aud") != expected_aud:
        return (
            f"identity token audience is {claims.get('aud')!r}, "
            f"expected {expected_aud!r}."
        )
    return None


def verify_gcp_oidc_token(expected_aud: str) -> None:
    """Read OIDC JWT from stdin, verify Google Cloud issuer and audience,
    and log structural claims."""
    claims = decode_jwt_claims(sys.stdin.read())
    reason = check_gcp_oidc_claims(claims, expected_aud)
    if reason:
        sys.exit(f"ERROR: {reason}")
    print("Verified GCP OIDC Token Structural Claims (redacted identity):")
    print(
        json.dumps(
            {
                "iss": claims.get("iss"),
                "aud": claims.get("aud"),
                "email_verified": claims.get("email_verified"),
                "has_email_claim": bool(claims.get("email")),
                "has_sub_claim": bool(claims.get("sub")),
            },
            indent=2,
        )
    )


def mint_pypi_token(index_base: str, project: str) -> None:
    """Read OIDC JWT from stdin and exchange it for a short-lived PyPI API token."""
    oidc_token = sys.stdin.read().strip()
    req = urllib.request.Request(
        f"{index_base}/_/oidc/mint-token",
        data=json.dumps({"token": oidc_token}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            err_json = json.loads(raw)
            sanitized = json.dumps(
                {"message": err_json.get("message"), "errors": err_json.get("errors")},
                indent=2,
            )
        except Exception:
            sanitized = raw
        print(f"Mint-token returned HTTP {e.code}:\n{sanitized}", file=sys.stderr)
        print(
            "\n======================================================================\n"
            f"NOTE: GCP OIDC token generation succeeded, but {index_base} rejected\n"
            f"the publisher lookup (HTTP {e.code}).\n"
            f"Ensure a Trusted/Pending Publisher is registered on "
            f"{index_base}/manage/account/publishing/\n"
            f"for PyPI Project Name {project!r} with this VM's GCP Service Account.\n"
            "======================================================================",
            file=sys.stderr,
        )
        sys.exit(1)

    token = body.get("token")
    if not token:
        sys.exit("ERROR: HTTP 200 from mint-token but no '.token' field found.")
    print(token)


COMMANDS = {
    "verify_built_wheel": verify_built_wheel,
    "verify_wheel_contents": verify_wheel_contents,
    "verify_publish_dist": verify_publish_dist,
    "fetch_oidc_audience": fetch_oidc_audience,
    "verify_gcp_oidc_token": verify_gcp_oidc_token,
    "mint_pypi_token": mint_pypi_token,
}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: wheel_utils.py <command> [args...]")
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        sys.exit(f"ERROR: unknown command '{cmd}'.")
    COMMANDS[cmd](*sys.argv[2:])


if __name__ == "__main__":
    main()
