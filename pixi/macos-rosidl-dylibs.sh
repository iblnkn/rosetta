#!/usr/bin/env bash
# macOS-only activation script, wired in via pixi.toml's osx-arm64 activation.
#
# rcl dlopens interface typesupport dylibs (lib*__rosidl_typesupport_*.dylib)
# by leaf name. install/setup.bash puts them on DYLD_LIBRARY_PATH, but SIP
# strips DYLD_* whenever a protected binary (/bin/sh, /usr/bin/env) execs, so
# node processes never see it and fail with "Could not load library". The env's
# lib/ is on librcl's rpath, so symlinking the built dylibs there makes them
# resolvable. Re-runs on every activation, so it self-heals after rebuilds.
_root="${PIXI_PROJECT_ROOT:-$PWD}"
if [ -n "${CONDA_PREFIX:-}" ]; then
    for _lib in "${_root}"/install/*/lib/*__rosidl_*.dylib; do
        [ -e "${_lib}" ] || continue
        _dest="${CONDA_PREFIX}/lib/$(basename "${_lib}")"
        # Only create/refresh symlinks; never clobber a real file the env ships.
        if [ ! -e "${_dest}" ] || [ -L "${_dest}" ]; then
            ln -sf "${_lib}" "${_dest}"
        fi
    done
    unset _lib _dest
fi
unset _root
