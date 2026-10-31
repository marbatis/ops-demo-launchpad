# Claude Code cloud Android environment

This directory preserves the personal **Android Builds** cloud setup tested on October 31, 2026. It makes the Android guidance, three Claude Code subagents, and a working Android build toolchain available before a fresh cloud session starts. It does not change the Python launchpad application.

## Files

- `android-agent-kit-cloud.zip`: the exact uploaded, self-contained account skill (18 files).
- `skill/android-agent-kit/`: the same ZIP contents as readable source, including adoption notes and the three active agent definitions in `templates/agents/`.
- `android-cloud-setup.sh`: the exact tested environment setup script. Its embedded archive contains that skill and the active agent definitions.
- `VERIFICATION.md`: actual fresh-session results and limits.

The bundled presentation references remain pinned to `2982ccfe100bb98ce87c50e385b3a6081d0c4a3c` from [PR #1](https://github.com/marbatis/ops-demo-launchpad/pull/1). Read the adoption notes before treating presentation examples as current APIs. New native projects default to Kotlin/Compose; existing projects retain their stack and use their own wrappers and compatible dependencies.

## Set up Claude Code cloud

1. Upload `android-agent-kit-cloud.zip` in Claude's skill customization page and enable `android-agent-kit`. This account skill is separate from the Mac's personal skill directory.
2. Create a personal cloud environment named **Android Builds** and upload `android-cloud-setup.sh` as its setup script.
3. Set these environment variables; no secret is required:

   ```text
   ANDROID_HOME=/opt/android-sdk
   ANDROID_SDK_ROOT=/opt/android-sdk
   ANDROID_AGENT_KIT_ROOT=/root/.claude/skills/android-agent-kit
   ```

4. Keep network access **Limited**, enable package-manager access, and allow these domains:

   ```text
   dl.google.com
   maven.google.com
   repo.maven.apache.org
   repo1.maven.org
   plugins.gradle.org
   plugins-artifacts.gradle.org
   services.gradle.org
   downloads.gradle.org
   github.com
   release-assets.githubusercontent.com
   developer.android.com
   developers.google.com
   source.android.com
   android.googlesource.com
   kotlinlang.org
   docs.gradle.org
   ```

5. Select **Android Builds** for new Android cloud tasks. For cloud project threads, choose it in the project's environment settings. Other environments do not inherit these tools or agents.

The setup targets fresh root Linux cloud environments, with the hosted OpenJDK 21 available. It installs the runtime skill under `$HOME/.claude/skills/android-agent-kit`, the three agents under `$HOME/.claude/agents`, and Android routing in `$HOME/.claude/CLAUDE.md`. It installs SDK platform 35, build-tools 35.0.0, platform-tools, and checksum-pinned Gradle 8.11.1. Install any additional SDK/JDK versions required by the actual project deliberately.

This is Marcelo's personal environment snapshot. It copies the Android SDK license acceptance already present in his local SDK; it does not accept new agreements or pipe `yes` into sdkmanager. Another user must review the applicable license terms and use their own accepted-license setup. A new or missing agreement causes setup to fail rather than be accepted automatically.

The environment preserves existing instructions and adds explicit commit/push restrictions plus Claude Bash deny rules for commits, pushes, PR creation and merges. These persistent denies were part of the tested configuration. A later authorized Git operation in this cloud environment may require deliberately adjusting those settings; a passing build is not authorization to publish or merge. This commit only stores the setup, and does not change that deployed permission configuration.

## Check a fresh session

Ask Claude to use the Android guidance and inspect `/opt/marbatis/android-cloud-setup.status` and the adjacent log. `READY` means prerequisites installed; it is not proof that an application built.

For an isolated environment probe, run outside every repository:

```bash
ANDROID_PROBE_DIR=$(mktemp -d /tmp/android-cloud-smoke.XXXXXX)
python3 /opt/marbatis/create_android_smoke.py "$ANDROID_PROBE_DIR"
cd "$ANDROID_PROBE_DIR"
gradle-8.11.1 wrapper --gradle-version 8.11.1 --distribution-type bin --gradle-distribution-sha256-sum f397b287023acdba1e9f6fc5ea72d22dd63669d59ed4a289a29b1a76eee151c6
./gradlew --no-daemon :app:assembleDebug :app:testDebugUnitTest :app:lintDebug
```

Invoke `android-release-validator`, `android-leak-hunter`, and `android-adaptive-reviewer` for scoped checks. Their presence as templates alone does not prove they ran. No emulator is provisioned: report runtime journeys, rendered UI and measured leak checks as unverified unless a suitable device is available.

Maven Central throttled downloads during the recorded run. Preserve Gradle caches, use bounded backoff for HTTP 429, and report exact failures. Do not disable TLS, bypass network policy, suppress checks, change dependency pins, or repeatedly refresh dependencies to hide a download problem.

## Maintenance

These are pinned snapshots. Editing readable source does not automatically rebuild the ZIP, replace the bootstrap's embedded archive, update the uploaded account skill, or update a saved cloud environment. Update those artifacts together, verify their content/checksums, and repeat fresh-session verification after material changes.

The bootstrap is for a fresh ephemeral environment, not migration of a customized developer machine. Identical pinned reruns are supported. A differing existing same-name agent aborts rather than being overwritten; skill extraction occurs before that conflict check.

[Claude skills](https://code.claude.com/docs/en/skills#use-skills-in-cowork-and-cloud-sessions) · [Cloud environments](https://code.claude.com/docs/en/cloud-environments) · [Subagents](https://code.claude.com/docs/en/sub-agents)
