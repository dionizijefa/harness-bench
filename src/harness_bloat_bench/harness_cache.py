"""Persistent host-side cache for pinned harness release artifacts."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from pathlib import PurePosixPath

from verifiers.v1.runtimes import DockerRuntime, Runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = REPO_ROOT / ".harness-cache"
CONTAINER_CACHE_ROOT = PurePosixPath("/var/cache/harness-bloat")


def cache_root() -> Path:
    override = os.environ.get("HARNESS_BLOAT_CACHE_DIR")
    return Path(override).expanduser().resolve() if override else DEFAULT_CACHE_ROOT


def linux_arch(machine: str | None = None) -> str:
    normalized = (machine or platform.machine()).lower()
    if normalized in {"aarch64", "arm64"}:
        return "arm64"
    if normalized in {"x86_64", "amd64", "x64"}:
        return "x64"
    raise ValueError(f"unsupported Linux harness architecture: {normalized}")


def omp_cache_path(version: str, arch: str | None = None) -> Path:
    normalized_version = version.removeprefix("v")
    normalized_arch = linux_arch(arch)
    return (
        cache_root()
        / "omp"
        / f"linux-{normalized_arch}"
        / normalized_version
        / "omp"
    )


def omp_release_url(version: str, arch: str | None = None) -> str:
    normalized_version = version.removeprefix("v")
    normalized_arch = linux_arch(arch)
    return (
        "https://github.com/can1357/oh-my-pi/releases/download/"
        f"v{normalized_version}/omp-linux-{normalized_arch}"
    )


def _download(url: str, destination: Path) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "harness-bloat-bench"})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
        expected = response.headers.get("Content-Length")
        with destination.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    if expected is not None and size != int(expected):
        raise OSError(f"downloaded {size} bytes, expected {expected}")
    if size == 0:
        raise OSError("downloaded an empty release artifact")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_tar_member(archive: Path, destination: Path, suffix: str) -> None:
    with tarfile.open(archive) as bundle:
        matches = [
            member for member in bundle.getmembers() if member.name.endswith(suffix)
        ]
        if len(matches) != 1:
            raise OSError(f"expected one {suffix!r} member, found {len(matches)}")
        source = bundle.extractfile(matches[0])
        if source is None:
            raise OSError(f"archive member is not a file: {matches[0].name}")
        with destination.open("wb") as output:
            shutil.copyfileobj(source, output)


def _ensure_release_file(
    *,
    harness: str,
    version: str,
    arch: str,
    filename: str,
    url: str,
    transform=None,
    expected_sha256: str | None = None,
    attempts: int = 5,
) -> Path:
    normalized_version = version.removeprefix("v")
    destination = (
        cache_root() / harness / f"linux-{arch}" / normalized_version / filename
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.is_file() and destination.stat().st_size > 0:
            return destination

        downloaded = destination.with_name(f".{filename}.{os.getpid()}.download")
        prepared = destination.with_name(f".{filename}.{os.getpid()}.prepared")
        last_error: Exception | None = None
        try:
            for attempt in range(1, attempts + 1):
                for temporary in (downloaded, prepared):
                    with contextlib.suppress(FileNotFoundError):
                        temporary.unlink()
                try:
                    _download(url, downloaded)
                    if transform is None:
                        os.replace(downloaded, prepared)
                    else:
                        transform(downloaded, prepared)
                    size = prepared.stat().st_size
                    digest = _sha256(prepared)
                    if size == 0:
                        raise OSError("prepared an empty release artifact")
                    if expected_sha256 is not None and digest != expected_sha256:
                        raise OSError(
                            f"checksum mismatch: expected {expected_sha256}, got {digest}"
                        )
                    prepared.chmod(0o755)
                    os.replace(prepared, destination)
                    destination.with_suffix(destination.suffix + ".json").write_text(
                        json.dumps(
                            {
                                "arch": arch,
                                "bytes": size,
                                "downloaded_at": dt.datetime.now(
                                    dt.timezone.utc
                                ).isoformat(),
                                "sha256": digest,
                                "url": url,
                                "version": normalized_version,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    return destination
                except Exception as error:
                    last_error = error
                    if attempt < attempts:
                        time.sleep(min(2 ** (attempt - 1), 16))
        finally:
            for temporary in (downloaded, prepared):
                with contextlib.suppress(FileNotFoundError):
                    temporary.unlink()

    raise RuntimeError(
        f"failed to cache {harness} {version} ({arch}) after {attempts} attempts: "
        f"{last_error}"
    )


def ensure_codex_cached(version: str, arch: str | None = None) -> Path:
    normalized_arch = linux_arch(arch)
    triple_arch = "aarch64" if normalized_arch == "arm64" else "x86_64"
    triple = f"{triple_arch}-unknown-linux-musl"
    normalized_version = version.removeprefix("v")
    return _ensure_release_file(
        harness="codex_agent",
        version=normalized_version,
        arch=normalized_arch,
        filename="codex",
        url=(
            "https://github.com/openai/codex/releases/download/"
            f"rust-v{normalized_version}/codex-{triple}.tar.gz"
        ),
        transform=lambda archive, output: _extract_tar_member(
            archive, output, f"codex-{triple}"
        ),
    )


def ensure_claude_cached(version: str, arch: str | None = None) -> Path:
    normalized_arch = linux_arch(arch)
    normalized_version = version.removeprefix("v")
    destination = (
        cache_root()
        / "claude-code"
        / f"linux-{normalized_arch}"
        / normalized_version
        / "claude"
    )
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    base = "https://downloads.claude.ai/claude-code-releases"
    manifest_url = f"{base}/{normalized_version}/manifest.json"
    with urllib.request.urlopen(manifest_url, timeout=60) as response:  # noqa: S310
        manifest = json.load(response)
    platform_name = f"linux-{normalized_arch}"
    checksum = manifest["platforms"][platform_name]["checksum"]
    return _ensure_release_file(
        harness="claude-code",
        version=normalized_version,
        arch=normalized_arch,
        filename="claude",
        url=f"{base}/{normalized_version}/{platform_name}/claude",
        expected_sha256=checksum,
    )


def ensure_opencode_cached(
    version: str,
    artifact_version: str,
    arch: str | None = None,
    *,
    musl: bool = False,
) -> Path:
    normalized_arch = linux_arch(arch)
    normalized_version = version.removeprefix("v")
    artifact_version = artifact_version.removeprefix("v")
    if normalized_arch == "arm64":
        package = "opencode-linux-arm64"
    elif artifact_version == "0.1.195":
        package = "opencode-linux-x64"
    else:
        package = "opencode-linux-x64-baseline"
    if musl:
        package += "-musl"
    cache_arch = f"{normalized_arch}{'-musl' if musl else ''}"
    return _ensure_release_file(
        harness="opencode",
        version=normalized_version,
        arch=cache_arch,
        filename="opencode",
        url=f"https://registry.npmjs.org/{package}/-/{package}-{artifact_version}.tgz",
        transform=lambda archive, output: _extract_tar_member(
            archive, output, "package/bin/opencode"
        ),
    )


def _ensure_release_tree(
    *,
    harness: str,
    version: str,
    arch: str,
    url: str,
    attempts: int = 5,
) -> Path:
    normalized_version = version.removeprefix("v")
    destination = cache_root() / harness / f"linux-{arch}" / normalized_version / "tree"
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / "tree.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.is_dir() and any(destination.iterdir()):
            return destination
        archive = destination.parent / f".{os.getpid()}.archive"
        unpacked = destination.parent / f".{os.getpid()}.unpacked"
        last_error: Exception | None = None
        try:
            for attempt in range(1, attempts + 1):
                with contextlib.suppress(FileNotFoundError):
                    archive.unlink()
                shutil.rmtree(unpacked, ignore_errors=True)
                try:
                    metadata = _download(url, archive)
                    unpacked.mkdir()
                    with tarfile.open(archive) as bundle:
                        bundle.extractall(unpacked, filter="data")
                    roots = list(unpacked.iterdir())
                    prepared = (
                        roots[0]
                        if len(roots) == 1 and roots[0].is_dir()
                        else unpacked
                    )
                    if destination.exists():
                        shutil.rmtree(destination)
                    os.replace(prepared, destination)
                    metadata.update(
                        {
                            "arch": arch,
                            "downloaded_at": dt.datetime.now(
                                dt.timezone.utc
                            ).isoformat(),
                            "url": url,
                            "version": normalized_version,
                        }
                    )
                    (destination.parent / "tree.json").write_text(
                        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    return destination
                except Exception as error:
                    last_error = error
                    if attempt < attempts:
                        time.sleep(min(2 ** (attempt - 1), 16))
        finally:
            with contextlib.suppress(FileNotFoundError):
                archive.unlink()
            shutil.rmtree(unpacked, ignore_errors=True)
    raise RuntimeError(
        f"failed to cache {harness} {version} ({arch}) after {attempts} attempts: "
        f"{last_error}"
    )


def ensure_docker_built_tree(
    *,
    harness: str,
    version: str,
    source_dir: str,
    install_script: str,
    arch: str | None = None,
    image: str = "python:3.11-slim-bullseye",
    bundle_python_runtime: bool = True,
) -> Path:
    """Build a composite harness once and cache its complete installed tree."""
    normalized_arch = linux_arch(arch)
    docker_platform = (
        "linux/arm64" if normalized_arch == "arm64" else "linux/amd64"
    )
    normalized_version = version.removeprefix("v")
    build_script = install_script.rstrip()
    if bundle_python_runtime:
        bundled_runtime = str(PurePosixPath(source_dir) / ".python-runtime")
        build_script += rf"""

# Virtual environments normally point back to the build image's interpreter.
# Bundle that interpreter and standard library so the cached tree remains
# runnable in Terminal-Bench task images that do not ship Python themselves.
PYTHON_RUNTIME={shlex.quote(bundled_runtime)}
rm -rf "$PYTHON_RUNTIME"
mkdir -p "$PYTHON_RUNTIME/bin" "$PYTHON_RUNTIME/lib"
cp -a /usr/local/bin/python3.11 "$PYTHON_RUNTIME/bin/"
cp -a /usr/local/lib/python3.11 "$PYTHON_RUNTIME/lib/"
for library in /usr/local/lib/libpython3.11.so*; do
    [ ! -e "$library" ] || cp -a "$library" "$PYTHON_RUNTIME/lib/"
done
# CPython extension modules can depend on distribution libraries that are not
# present, or have a different SONAME, in arbitrary Terminal-Bench images.
# Bundle those non-glibc shared objects alongside the interpreter.
find /usr/local/lib/python3.11 -type f -name '*.so' -exec ldd {{}} \; 2>/dev/null |
    awk '/=> \/[^ ]+/ {{ print $3 }} /^\/[[:graph:]]+ \(/ {{ print $1 }}' |
    sort -u |
    while IFS= read -r library; do
        case "$(basename "$library")" in
            ld-linux*|libc.so.*|libdl.so.*|libm.so.*|libpthread.so.*|libresolv.so.*|librt.so.*|libutil.so.*)
                continue
                ;;
        esac
        case "$library" in
            /lib/*|/usr/lib/*)
                cp -L "$library" "$PYTHON_RUNTIME/lib/$(basename "$library")"
                ;;
        esac
    done
"""
    recipe = {
        "arch": normalized_arch,
        "image": image,
        "install_script": build_script,
        "source_dir": source_dir,
        "version": normalized_version,
    }
    recipe_sha256 = hashlib.sha256(
        json.dumps(recipe, sort_keys=True).encode()
    ).hexdigest()
    destination = (
        cache_root()
        / harness
        / f"linux-{normalized_arch}"
        / normalized_version
        / "tree"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / "tree.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.is_dir() and any(destination.iterdir()):
            try:
                metadata = json.loads(
                    (destination.parent / "tree.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                metadata = {}
            if metadata.get("recipe_sha256") == recipe_sha256:
                return destination
            shutil.rmtree(destination)
        container = f"harness-cache-{harness}-{os.getpid()}"
        temporary = destination.parent / f".{os.getpid()}.tree"
        shutil.rmtree(temporary, ignore_errors=True)
        try:
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--platform",
                    docker_platform,
                    "--network",
                    "host",
                    "--name",
                    container,
                    "--entrypoint",
                    "sleep",
                    image,
                    "infinity",
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["docker", "exec", "-i", container, "sh", "-c", build_script],
                check=True,
            )
            subprocess.run(
                ["docker", "cp", f"{container}:{source_dir}", str(temporary)],
                check=True,
                capture_output=True,
            )
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(temporary, destination)
            (destination.parent / "tree.json").write_text(
                json.dumps(
                    {
                        "arch": normalized_arch,
                        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "image": image,
                        "recipe_sha256": recipe_sha256,
                        "version": normalized_version,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return destination
        finally:
            subprocess.run(
                ["docker", "rm", "--force", container],
                check=False,
                capture_output=True,
            )
            shutil.rmtree(temporary, ignore_errors=True)


def install_cached_python_runtime_script(
    source_dir: str, dependency_dir: str
) -> str:
    """Return an offline script that restores a composite bundle's Python.

    The cached interpreter may need older shared-library SONAMEs than the task
    image provides. Keep those libraries private to the harness process: copying
    them into ``/usr/local/lib`` can shadow the task image's own libraries and
    break unrelated commands, including Harbor's verifier bootstrap.
    """
    bundled_runtime = str(PurePosixPath(source_dir) / ".python-runtime")
    return f"""\
PYTHON_RUNTIME={shlex.quote(bundled_runtime)}
PYTHON_DEPENDENCY_DIR={shlex.quote(dependency_dir)}
test -x "$PYTHON_RUNTIME/bin/python3.11"
mkdir -p /usr/local/bin /usr/local/lib
cp -a "$PYTHON_RUNTIME/bin/python3.11" /usr/local/bin/python3.11
cp -a "$PYTHON_RUNTIME/lib/python3.11" /usr/local/lib/
rm -rf "$PYTHON_DEPENDENCY_DIR"
mkdir -p "$PYTHON_DEPENDENCY_DIR"
for library in "$PYTHON_RUNTIME"/lib/*; do
    [ -f "$library" ] || continue
    name="$(basename "$library")"
    if command -v ldconfig >/dev/null 2>&1 && \
        ldconfig -p 2>/dev/null | awk '{{print $1}}' | grep -Fqx "$name"; then
        continue
    fi
    found=false
    for system_library in /lib/*/"$name" /usr/lib/*/"$name"; do
        if [ -e "$system_library" ]; then
            found=true
            break
        fi
    done
    [ "$found" = true ] || cp -a "$library" "$PYTHON_DEPENDENCY_DIR/$name"
done
"""


def ensure_omp_cached(
    version: str,
    arch: str | None = None,
    *,
    attempts: int = 5,
) -> Path:
    """Download one OMP artifact once, safely across concurrent rollout processes."""
    normalized_arch = linux_arch(arch)
    destination = omp_cache_path(version, normalized_arch)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(".lock")
    url = omp_release_url(version, normalized_arch)

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.is_file() and destination.stat().st_size > 0:
            return destination

        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        last_error: Exception | None = None
        try:
            for attempt in range(1, attempts + 1):
                with contextlib.suppress(FileNotFoundError):
                    temporary.unlink()
                try:
                    metadata = _download(url, temporary)
                    temporary.chmod(0o755)
                    os.replace(temporary, destination)
                    metadata.update(
                        {
                            "arch": normalized_arch,
                            "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "url": url,
                            "version": version.removeprefix("v"),
                        }
                    )
                    destination.with_suffix(".json").write_text(
                        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    return destination
                except Exception as error:  # retry transient HTTP and filesystem errors
                    last_error = error
                    if attempt < attempts:
                        time.sleep(min(2 ** (attempt - 1), 16))
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()

    raise RuntimeError(
        f"failed to cache OMP {version} ({normalized_arch}) after {attempts} attempts: "
        f"{last_error}"
    )


async def stage_cached_file(runtime: Runtime, source: Path, destination: str) -> None:
    """Copy a cached artifact into a runtime without buffering Docker copies in RAM."""
    if isinstance(runtime, DockerRuntime):
        parent = str(PurePosixPath(destination).parent)
        result = await runtime.run(["mkdir", "-p", parent], {})
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"failed to create runtime cache directory: {detail}")
        if getattr(runtime, "_harness_cache_mounted", False):
            relative = source.resolve().relative_to(cache_root())
            mounted_source = str(CONTAINER_CACHE_ROOT / relative.as_posix())
            link = await runtime.run(
                ["ln", "-sfn", mounted_source, destination], {}
            )
            if link.exit_code != 0:
                detail = (link.stderr or link.stdout).strip()
                raise RuntimeError(f"failed to link cached artifact: {detail}")
            return
        container = runtime.info.id
        process = await asyncio.create_subprocess_exec(
            "docker",
            "cp",
            str(source),
            f"{container}:{destination}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"failed to copy cached artifact into Docker: "
                f"{stderr.decode(errors='replace').strip()}"
            )
        return

    await runtime.write(destination, await asyncio.to_thread(source.read_bytes))


async def stage_cached_tree(runtime: Runtime, source: Path, destination: str) -> None:
    """Copy a cached installation tree into a runtime."""
    if isinstance(runtime, DockerRuntime) and getattr(
        runtime, "_harness_cache_mounted", False
    ):
        relative = source.resolve().relative_to(cache_root())
        mounted_source = str(CONTAINER_CACHE_ROOT / relative.as_posix())
        parent = str(PurePosixPath(destination).parent)
        link = await runtime.run(
            [
                "sh",
                "-c",
                f"mkdir -p {shlex.quote(parent)} && "
                f"rm -rf {shlex.quote(destination)} && "
                f"ln -s {shlex.quote(mounted_source)} {shlex.quote(destination)}",
            ],
            {},
        )
        if link.exit_code != 0:
            detail = (link.stderr or link.stdout).strip()
            raise RuntimeError(f"failed to link cached tree: {detail}")
        return
    result = await runtime.run(["mkdir", "-p", destination], {})
    if result.exit_code != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"failed to create runtime cache directory: {detail}")
    if isinstance(runtime, DockerRuntime):
        container = runtime.info.id
        process = await asyncio.create_subprocess_exec(
            "docker",
            "cp",
            f"{source}/.",
            f"{container}:{destination}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"failed to copy cached tree into Docker: "
                f"{stderr.decode(errors='replace').strip()}"
            )
        return

    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive_file:
        with tarfile.open(archive_file.name, "w:gz") as bundle:
            for child in source.iterdir():
                bundle.add(child, arcname=child.name)
        archive = f"{destination}/.harness-cache.tar.gz"
        data = await asyncio.to_thread(Path(archive_file.name).read_bytes)
        await runtime.write(archive, data)
        extract = await runtime.run(
            ["sh", "-c", f"tar -xzf {archive} -C {destination} && rm -f {archive}"],
            {},
        )
        if extract.exit_code != 0:
            detail = (extract.stderr or extract.stdout).strip()
            raise RuntimeError(f"failed to extract cached runtime tree: {detail}")
