---
status: live
topic: vamp-interface
---

# NDI 6 Python bindings — compatibility research

Date: 2026-05-05. Question: cyndilib `Receiver.receive(recv_video, ...)` segfaults against libndi.so.6 (NDI SDK 6.3.2). What are the fix paths?

## TL;DR

- **cyndilib *is* the NDI 6 wrapper.** PyPI 0.1.1 (2026-03-22) is built against NDI SDK 6.x — the SDK was bumped to 6.0.0 in cyndilib 0.0.3 and to 6.1.1 in 0.0.6. The "0.1.1 == NDI 5" assumption was wrong. Repo is actively maintained (commits within the last week). Latest commits are CI/lockfile maintenance, no NDI 6.3-specific fixes.
- **`ndi-python` (buresu) is dead** for our purposes: setup.py pins `version='5.1.1.5'`, last source commit 2022-06-14 (Python 3.7–3.10), explicit NDI SDK 5.x. Repo `pushed_at` is recent only because of automated stuff. Not a viable NDI 6 replacement.
- **NDI 5 ↔ NDI 6 wire protocol is bidirectionally compatible** per Vizrt's "NDI is forward and backward compatible" statement; an NDI 5 receiver decodes NDI 6 high-bandwidth/HX2/HX3 streams (HDR/10-bit features degrade gracefully — non-HDR receivers just see SDR).
- **NDI SDK 5 Linux tarball:** historically at `https://downloads.ndi.tv/SDK/NDI_SDK_Linux/Install_NDI_SDK_v5_Linux.tar.gz`. Vizrt's current download portal only lists 6.x and gates downloads behind a per-user form. The Arch AUR `ndi-sdk` package and Internet Archive both have v5 installers preserved. No primary GitHub mirror by Vizrt.

## Evidence

### cyndilib NDI version targeting

GitHub releases (`gh api`):

| Release | Date | NDI SDK |
|---------|------|---------|
| 0.1.1 | 2026-03-22 | (carried forward, 6.x) |
| 0.0.6 | 2024-06-01 | 6.1.1 (explicit bump) |
| 0.0.3 | 2024-07-07 | 6.0.0 (explicit bump) |

Recent commit activity on `nocarryr/cyndilib` (2026-04-22 → 2026-05-04, lockfile + CI runner work). Repo is alive.

### Open issues — no segfault report yet

`gh issue list --repo nocarryr/cyndilib --state all` returned 10 issues. None mention "segfault", "SIGSEGV", "libndi.so.6", or `recv_video` crashes. Open issues are: #36 dependency dashboard, #26 audio-forwarding bug, #14 signal delay, #5 numpy compatibility. **Our crash is not a known-public issue → file one.** Build a minimal repro (Finder → 1 source → Receiver.connect → single recv_video call → backtrace via `gdb python`).

### ndi-python (buresu)

- `setup.py` pins NDI SDK 5.x branding (`version='5.1.1.5'`).
- Last meaningful source commit: 2022-06-14.
- `pushed_at: 2026-04-24` is misleading; recent activity is automated.
- Not a viable upgrade path; no NDI 6 fork found in search.

### Wire-protocol compatibility (NDI 5 ↔ 6)

Vizrt blog/docs: "all versions of NDI are forward and backward compatible." NDI 6 adds optional HDR + 10-bit; non-HDR receivers fall back to SDR. iOS NDI HX Camera senders should decode fine on a libndi.so.5 receiver.

### NDI SDK 5 Linux availability

- Official portal (ndi.video/for-developers/ndi-sdk/download/): only 6.3.x, request-gated, per-user URLs.
- Historical direct URL `downloads.ndi.tv/SDK/NDI_SDK_Linux/Install_NDI_SDK_v5_Linux.tar.gz` was canonical and may still resolve.
- Arch User Repository `ndi-sdk` package and `archive.org/details/ndi-5-sdk-for-i-os-and-mac-os` mirror the v5 installer (iOS/Mac archive; Linux mirrors exist on AUR PKGBUILD sources).
- No first-party "older versions" page on ndi.video.

## Recommendations

In order of preference:

1. **Diagnose against cyndilib 0.1.1 first.** Get a gdb backtrace of the segfault (`gdb --args python repro.py`, then `bt full`). Likely culprits: (a) Finder thread races (the new NDI 6.3 discovery has caused issues elsewhere), (b) Receiver creation before Finder yields a valid `Source` ptr, (c) frame-type mask bits — try `recv_video | recv_status_change` or constrain to a specific colour-format. File an issue on `nocarryr/cyndilib` with the trace; maintainer is responsive (lockfile commits weekly).
2. **Pin libndi to 5.x as a fallback.** Drop `libndi.so.5` next to your binary or set `LD_LIBRARY_PATH`; cyndilib loads via `ctypes`/Cython linking against the NDI shim, and NDI 5 will decode the iOS NDI HX Camera stream. Source the v5 SDK from the AUR PKGBUILD URL or the Internet Archive — these are not "sketchy mirrors," they're the same tarball Arch packagers have used for years. Verify SHA against any preserved release notes.
3. **Do not switch to `ndi-python`.** It's NDI 5 frozen at 2022; any segfault risk you have today persists there with worse maintenance.
4. **Last resort: ctypes shim over libndi.so.6.** The NDI C API is small (~20 functions you'd touch). A 200-line ctypes wrapper around `NDIlib_recv_create_v3` + `NDIlib_recv_capture_v3` sidesteps Cython binding bugs entirely. Worth it only if cyndilib upstream is unresponsive for >1 week.

## Concrete URLs verified

- https://github.com/nocarryr/cyndilib (active, last push 2026-05-04)
- https://github.com/buresu/ndi-python (frozen 2022, NDI SDK 5)
- https://pypi.org/project/cyndilib/ (0.1.1, 2026-03-22, Python ≥3.10)
- https://ndi.video/for-developers/ndi-sdk/download/ (6.3 only, gated)
- https://aur.archlinux.org/packages/ndi-sdk (v5 PKGBUILD reference)
- https://archive.org/download/ndi-5-sdk-for-i-os-and-mac-os (Internet Archive mirror, iOS/Mac variant)
