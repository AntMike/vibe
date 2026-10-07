use super::CommandError;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// Long enough for a CLI agent to start, sign in and read an hour of transcript.
const TIMEOUT: Duration = Duration::from_secs(300);

/// Run a command-line AI (`claude -p`, `codex exec -`, `gemini`, or the user's own) through the
/// shell, so it is found on PATH and runs on the user's subscription without an API key. The
/// prompt goes in on stdin, never into the command line, so transcript text can't reach the shell.
/// It runs in the temp folder so an agent doesn't pick up a project around Vibe's working directory.
#[tauri::command]
pub async fn ask_cli(command: String, prompt: String) -> Result<String, CommandError> {
    let fail = |message: String| CommandError {
        code: "ai_cli_failed".to_string(),
        message,
    };
    if command.trim().is_empty() {
        return Err(fail("no command set".to_string()));
    }

    let mut child = shell(&command)
        .current_dir(std::env::temp_dir())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // ponytail: kills the shell, not a CLI it started; a process group/job object if strays pile up.
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| fail(format!("could not start `{command}`: {error}")))?;

    // Written alongside the read: a CLI that answers before draining stdin would otherwise block us both.
    let mut stdin = child.stdin.take().expect("stdin is piped");
    let writer = tokio::spawn(async move {
        let _ = stdin.write_all(prompt.as_bytes()).await;
    });
    let output = tokio::time::timeout(TIMEOUT, child.wait_with_output())
        .await
        .map_err(|_| fail(format!("`{command}` did not answer within {} seconds", TIMEOUT.as_secs())))?
        .map_err(|error| fail(format!("`{command}` failed: {error}")))?;
    let _ = writer.await;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !output.status.success() || stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        // The last 500 characters, where a CLI says what went wrong.
        let tail = stderr.char_indices().rev().nth(499).map_or(0, |(i, _)| i);
        return Err(fail(format!(
            "`{command}` exited with {}: {}",
            output.status,
            &stderr[tail..]
        )));
    }
    Ok(stdout)
}

#[cfg(windows)]
fn shell(command: &str) -> Command {
    // npm installs CLIs as .cmd shims, which only cmd resolves.
    let mut cmd = Command::new("cmd");
    cmd.arg("/D").arg("/S").arg("/C").raw_arg(format!("\"{command}\""));
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    cmd
}

#[cfg(not(windows))]
fn shell(command: &str) -> Command {
    // A login shell: apps started from the Dock don't inherit the PATH a terminal has.
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    let mut cmd = Command::new(shell);
    cmd.arg("-lc").arg(command);
    cmd
}

#[cfg(test)]
mod tests {
    use super::ask_cli;

    #[tokio::test]
    async fn pipes_the_prompt_through_the_shell_and_reports_failures() {
        // A quoted argument checks the command reaches the shell intact; the Cyrillic, that stdin is UTF-8.
        let echo = if cfg!(windows) { r#"findstr "^""# } else { "cat" };
        assert_eq!(
            ask_cli(echo.to_string(), "Олексій, Kubernetes".to_string()).await.unwrap(),
            "Олексій, Kubernetes"
        );
        let error = ask_cli("vibe-no-such-cli".to_string(), "x".to_string()).await.unwrap_err();
        assert_eq!(error.code, "ai_cli_failed");
    }
}
