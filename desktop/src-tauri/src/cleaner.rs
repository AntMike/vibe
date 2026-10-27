use crate::error::LogError;
use crate::ffmpeg::get_vibe_temp_folder;
use crate::{cmd::app::get_logs_folder, config, logging::get_log_path};
use eyre::{eyre, ContextCompat, Result};

pub fn clean_old_logs(app: &tauri::AppHandle) -> Result<()> {
    tracing::debug!("clean old logs");
    let current_log_path = get_log_path(&app.clone())?;

    // Get logs folder
    let logs_folder = get_logs_folder(app.to_owned())?;
    let logs_folder = logs_folder.to_str().context("tostr")?;

    // Remove suffix
    let logs_folder = logs_folder.strip_suffix('/').unwrap_or(logs_folder);
    let logs_folder = logs_folder.strip_suffix('\\').unwrap_or(logs_folder);
    let pattern = format!(
        "{}/{}*{}",
        logs_folder,
        config::LOG_FILENAME_PREFIX,
        config::LOG_FILENAME_SUFFIX
    );
    tracing::debug!("searching old logs in {}", pattern);
    for path in glob::glob(&pattern)? {
        let path = path?;
        if path == current_log_path {
            tracing::debug!("Skip clean of current log path {}", path.display());
            continue;
        }
        tracing::debug!("clean old log {}", path.display());
        std::fs::remove_file(path)?;
    }
    Ok(())
}

pub fn clean_old_files() -> Result<()> {
    let current_temp_dir = get_vibe_temp_folder();
    let temp_dir = std::env::temp_dir();
    let temp_dir = temp_dir.to_str().unwrap_or_default();
    // Remove suffix
    let temp_dir = temp_dir.strip_suffix('/').unwrap_or(temp_dir);
    let temp_dir = temp_dir.strip_suffix('\\').unwrap_or(temp_dir);
    let pattern = format!("{}/vibe_temp*", temp_dir);
    tracing::debug!("searching old files in {}", pattern);
    for path in glob::glob(&pattern)? {
        let path = path?;
        if path == current_temp_dir {
            tracing::debug!("Skip deletion of {}", current_temp_dir.display());
            continue;
        }
        tracing::debug!("Clean old folder {}", path.clone().display());
        std::fs::remove_dir_all(path.clone())
            .map_err(|e| eyre!("failed to delete {}: {:?}", path.display(), e))
            .log_error();
    }
    Ok(())
}

pub fn clean_updater_files() -> Result<()> {
    let current_temp_dir = get_vibe_temp_folder();
    let temp_dir = std::env::temp_dir();
    let temp_dir = temp_dir.to_str().unwrap_or_default();
    // Remove suffix
    let temp_dir = temp_dir.strip_suffix('/').unwrap_or(temp_dir);
    let temp_dir = temp_dir.strip_suffix('\\').unwrap_or(temp_dir);
    let pattern = format!("{}/vibe*-updater*", temp_dir);
    tracing::debug!("searching old files in {}", pattern);
    for path in glob::glob(&pattern)? {
        let path = path?;
        if path == current_temp_dir {
            tracing::debug!("Skip deletion of {}", current_temp_dir.display());
            continue;
        }
        if path.is_dir() {
            tracing::debug!("Clean old folder {}", path.display());
            std::fs::remove_dir_all(&path)
                .map_err(|e| eyre!("failed to delete {}: {:?}", path.display(), e))
                .log_error();
        } else {
            tracing::debug!("Skipping non-directory path {}", path.display());
        }
    }
    Ok(())
}

/// Delete kept `mic.flac` / `call.flac` tracks older than `days` days from the direct
/// subfolders of `projects_path`. Returns how many files were deleted.
#[tauri::command]
pub fn shrink_recordings(projects_path: String, days: u32) -> Result<u32> {
    if days == 0 {
        return Ok(0);
    }
    let Some(cutoff) = std::time::SystemTime::now().checked_sub(std::time::Duration::from_secs(u64::from(days) * 86400))
    else {
        return Ok(0);
    };
    let Ok(folders) = std::fs::read_dir(&projects_path) else {
        return Ok(0);
    };
    let mut deleted = 0;
    for folder in folders.flatten().filter(|entry| entry.path().is_dir()) {
        for name in ["mic.flac", "call.flac"] {
            let path = folder.path().join(name);
            let old = std::fs::metadata(&path)
                .and_then(|meta| meta.modified())
                .is_ok_and(|modified| modified < cutoff);
            if old {
                match std::fs::remove_file(&path) {
                    Ok(()) => deleted += 1,
                    Err(e) => tracing::warn!("failed to delete {}: {:?}", path.display(), e),
                }
            }
        }
    }
    Ok(deleted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    fn touch(path: &std::path::Path, age_days: u64) {
        let file = std::fs::File::create(path).unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(age_days * 86400)).unwrap();
    }

    #[test]
    fn shrink_recordings_deletes_only_old_tracks() {
        let root = std::env::temp_dir().join(format!("vibe_shrink_{}", crate::ffmpeg::random_string(10)));
        let project = root.join("project");
        std::fs::create_dir_all(&project).unwrap();
        touch(&project.join("mic.flac"), 10);
        touch(&project.join("call.flac"), 1);
        touch(&project.join("audio.wav"), 10);
        let root_str = root.to_string_lossy().to_string();

        assert_eq!(shrink_recordings(root_str.clone(), 0).unwrap(), 0);
        assert!(project.join("mic.flac").exists());

        assert_eq!(shrink_recordings(root_str, 5).unwrap(), 1);
        assert!(!project.join("mic.flac").exists());
        assert!(project.join("call.flac").exists());
        assert!(project.join("audio.wav").exists());
        assert_eq!(shrink_recordings(root.join("missing").to_string_lossy().to_string(), 5).unwrap(), 0);

        std::fs::remove_dir_all(&root).ok();
    }
}
