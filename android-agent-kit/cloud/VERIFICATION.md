# Fresh-session verification — October 31, 2026

A fresh Claude Code cloud session used the saved **Android Builds** environment with no repository selected. The Android skill was selected automatically, and all three registered Android subagents were actually invoked.

| Check | Observed result |
|---|---|
| Setup marker | `READY` |
| Java | OpenJDK 21.0.12.1 |
| Gradle | 8.11.1 |
| Android SDK | Platform 35, build-tools 35.0.0, platform-tools |
| Debug APK | Built, 10,073,342 bytes, Android Debug signature |
| JVM test | 1 test, 0 failures/errors/skips |
| Lint | 0 errors, 4 warnings; no baseline or suppressions |
| Release validator | Independently verified APK size/hash/signature, JUnit/lint counts and Gradle checksums; no setup blockers |
| Leak hunter | Static review completed; no retention risks found; no edits or runtime heap evidence |
| Adaptive reviewer | Static review completed; identified inset/status-bar/theme limitations in the minimal probe; no edits |
| Isolation | Original working directory empty and unchanged; scratch app outside repositories with no `.git` |
| Cleanup | Probe-started Gradle daemon and adb server stopped normally; tools/cache/artifacts retained |

The command `./gradlew --no-daemon :app:assembleDebug :app:testDebugUnitTest :app:lintDebug` exited 0 on its third attempt. Earlier attempts hit Maven Central HTTP 429s. The successful attempt reported 51 tasks: 7 executed and 44 up to date. The wrapper succeeded on its third retry after an initial failed invocation. No dependency pins, mirror, proxy protections or checks were changed.

The probe pins AGP 8.9.2, Kotlin/Compose compiler 2.1.20, Compose BOM 2025.04.01 and Activity Compose 1.10.1. These are a reproducible verification baseline, not defaults mandated for future apps. The test confirms the JVM runner works, not application behavior.

The lint warnings were two `GradleDependency` notices for the intentional pins, one `DataExtractionRules`, and one `MissingApplicationIcon`. The sample is a build probe, not a production-ready app.

No emulator/device was provisioned. Install/launch journeys, rendered previews, TalkBack, instrumented tests, measured leaks, release/R8 and release-signing builds were not verified. Because the build reused the cache populated by retries, this does not establish a retry-free build with an empty dependency cache. No commit, push, PR, app upload or publication occurred during that validation session; the later repository commit is a separate authorized operation.

## Artifact checksums

| Artifact | SHA-256 |
|---|---|
| `android-agent-kit-cloud.zip` | `9ec01d655c3d8e51224b5f8fcb8ed55d9d4b385f9b5ffd80f84c416f4664e540` |
| `android-cloud-setup.sh` | `cc302e73500d1cd26385b800727acdcd621a8a627f95a78d131945fae920feb0` |
| Probe APK | `842ad44c6e5f00a967fa7241f03ebd192d3eb6df1376fbe49f335ecaeb699fcc` |

The APK, logs, JUnit XML and lint reports were retained in the ephemeral cloud session, not added to this repository. Account-specific URLs, screenshots, machine paths and local configuration are excluded from this package.
