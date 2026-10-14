// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod cleaner;
mod cli;
mod cmd;
mod config;
mod config_watcher;
mod diagnostics;
mod dictation_indicator;
mod error;
mod ffmpeg;
mod handoff;
mod keepawake;
mod logging;
mod meeting_prompt;
mod notifications;
mod server;
mod setup;
mod transcript;
mod tray;
mod voice_activity;
use tauri::Emitter;

#[cfg(target_os = "macos")]
mod dock;

#[cfg(windows)]
mod call_capture;
#[cfg(windows)]
mod custom_protocol;

use eyre::{eyre, Result};
use tauri::Manager;
use tauri_plugin_window_state::StateFlags;

use error::LogError;

#[tokio::main]
async fn main() -> Result<()> {
    // Attach console in Windows:
    #[cfg(all(windows, not(debug_assertions)))]
    cli::attach_console();

    #[allow(unused_mut)]
    let mut builder = tauri::Builder::default()
        .manage(tray::TrayState::default())
        .manage(keepawake::KeepAwake::new())
        .manage(tokio::sync::Mutex::<Option<handoff::HandoffState>>::new(None))
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_single_instance::init(|app, argv, cwd| {
            tracing::debug!("{}, {argv:?}, {cwd}", app.package_info().name);
            tray::show_main_window(app);
            app.emit("single-instance", argv).map_err(|e| eyre!("{:?}", e)).log_error();
        }))
        .setup(|app| {
            setup::setup(app)?;
            Ok(())
        })
        .plugin(
            tauri_plugin_window_state::Builder::default()
                .with_state_flags(!StateFlags::VISIBLE)
                .with_denylist(&["meeting-prompt"])
                .build(),
        )
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_deep_link::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            Some(vec![cli::AUTOSTART_ARG]),
        ));

    #[cfg(target_os = "macos")]
    {
        builder = builder.plugin(tauri_nspanel::init());
    }

    let app = builder
        .invoke_handler(tauri::generate_handler![
            cmd::download::download_file,
            cmd::handoff_cmd::handoff_status,
            cmd::handoff_cmd::handoff_start,
            cmd::handoff_cmd::handoff_stop,
            cmd::handoff_cmd::handoff_regenerate_token,
            cmd::handoff_cmd::handoff_revoke_device,
            cmd::handoff_cmd::handoff_revoke_all,
            cmd::keepawake_cmd::keepawake_start,
            cmd::keepawake_cmd::keepawake_stop,
            cmd::app::get_cargo_features,
            cmd::config::write_config_atomically,
            cmd::config::get_config_path,
            cmd::skill::install_agent_skill,
            cmd::skill::get_agent_paths,
            tray::set_tray,
            cmd::transcribe::transcribe,
            cmd::ai_cli::ask_cli,
            cmd::files::glob_files,
            cmd::files::pick_media_paths,
            cmd::download::download_model,
            cmd::download::check_model_files,
            cmd::download::cleanup_partial_downloads,
            cmd::server_cmd::load_model,
            cmd::server_cmd::get_gpu_devices,
            cmd::server_cmd::get_model_metadata,
            cmd::server_cmd::get_api_base_url,
            cmd::server_cmd::start_api_server,
            cmd::server_cmd::stop_api_server,
            cmd::app::get_commit_hash,
            cmd::app::is_avx2_enabled,
            cmd::app::is_online,
            cmd::files::get_path_dst,
            cmd::app::get_logs,
            cmd::files::open_path,
            cmd::files::get_save_path,
            cmd::files::get_argv,
            cmd::files::get_default_projects_path,
            cmd::audio::get_audio_devices,
            cmd::audio::start_record,
            cmd::app::get_models_folder,
            cmd::app::get_logs_folder,
            cmd::app::show_log_path,
            cmd::app::show_temp_path,
            cmd::files::get_ffmpeg_path,
            cmd::ytdlp::download_audio,
            cmd::ytdlp::get_temp_path,
            cmd::ytdlp::get_latest_ytdlp_version,
            cmd::app::is_crashed_recently,
            cmd::app::rename_crash_file,
            cmd::app::type_text,
            cmd::permissions::request_system_audio_permission,
            cmd::permissions::open_system_audio_settings,
            cmd::permissions::get_system_audio_permission_status,
            cmd::permissions::get_microphone_permission_status,
            cmd::permissions::request_microphone_permission,
            cmd::permissions::open_microphone_settings,
            cmd::permissions::get_screen_recording_permission_status,
            cmd::permissions::request_screen_recording_permission,
            cmd::permissions::open_screen_recording_settings,
            dictation_indicator::get_dictation_indicator_enabled,
            dictation_indicator::set_dictation_indicator_enabled,
            dictation_indicator::show_dictation_indicator,
            dictation_indicator::get_dictation_indicator_state,
            dictation_indicator::dictation_indicator_ready,
            dictation_indicator::hide_dictation_indicator,
            meeting_prompt::get_meeting_detection_enabled,
            meeting_prompt::set_meeting_detection_enabled,
            meeting_prompt::get_meeting_prompt_state,
            meeting_prompt::dismiss_meeting_prompt,
            meeting_prompt::meeting_prompt_ready,
            meeting_prompt::test_meeting_notification,
            notifications::set_notification_labels,
            notifications::show_slack_send_toast
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app, event| match event {
        // Clicking the dock icon while the window is hidden in the tray has to bring it back;
        // macOS reports the click here rather than as a window event.
        #[cfg(target_os = "macos")]
        tauri::RunEvent::Reopen { has_visible_windows, .. } => {
            if !has_visible_windows {
                tray::show_main_window(app);
            }
        }
        tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit => {
            // A base URL left in the config file would point an agent at a dead port.
            cmd::config::set_api_base_url(app, None).log_error();
            let mutex = app.state::<tokio::sync::Mutex<setup::ServerState>>();
            if let Ok(mut guard) = mutex.try_lock() {
                if let Some(ref mut process) = guard.process {
                    process.kill();
                }
            };
            // Drop the handoff router so the iroh endpoint closes cleanly.
            let handoff = app.state::<tokio::sync::Mutex<Option<handoff::HandoffState>>>();
            if let Ok(mut guard) = handoff.try_lock() {
                guard.take();
            };
        }
        _ => {}
    });
    Ok(())
}
