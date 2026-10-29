#!/usr/bin/env bash
# macOS-only linker wrapper, wired in via pixi.toml's osx-arm64 LDFLAGS.
#
# robostack-jazzy's clang (conda-forge, cross-compiling "arm64-apple-darwin20.0.0"
# triple) ships its own bundled `ld`. On recent macOS SDKs (observed on macOS 26.5)
# that bundled linker fails to resolve implicit libSystem symbols
# (___assert_rtn, ___stack_chk_fail, ___stack_chk_guard), breaking every dylib
# link in ament_cmake packages (e.g. rosidl-generated *_generator_c libraries).
#
# Fix: use the system linker instead (matches the installed SDK by construction).
# One fixup: conda's clang passes `-lto_library <its versioned libLTO>`, which
# Apple's ld rejects (basename must be exactly `libLTO.dylib`) — and conda
# LLVM's libLTO wouldn't match Apple ld's LTO API anyway. The workspace doesn't
# build with LTO, so drop the pair; Apple's ld falls back to its own libLTO if
# LTO objects ever appear.
set -euo pipefail

real_ld="$(xcrun -f ld)"

args=()
skip_next=0
for arg in "$@"; do
  if [ "$skip_next" = 1 ]; then
    skip_next=0
    continue
  fi
  if [ "$arg" = "-lto_library" ]; then
    skip_next=1
    continue
  fi
  args+=("$arg")
done

exec "$real_ld" "${args[@]}"
