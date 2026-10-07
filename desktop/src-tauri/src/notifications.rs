//! What the user chose under Settings → Notifications, read straight from the config file, plus the
//! labels the frontend hands over (Rust has no translations) and the Windows toast for a meeting.

use crate::config::STORE_FILENAME;
use crate::error::LogError;
use meeting_detect::Source;
use serde::Deserialize;
use std::sync::Mutex;
use tauri::{AppHandle, Manager};
use tauri_plugin_notification::NotificationExt;
use tauri_plugin_store::StoreExt;

/// `"popup"` (Vibe's own window, the default) or `"system"` (a Windows notification).
const MEETING_STYLE_KEY: &str = "notifications.meetingStyle";
/// How long the meeting alert stays up; 0 means until it is dismissed or the call ends.
const MEETING_SECONDS_KEY: &str = "notifications.meetingSeconds";
const MEETING_SOUND_KEY: &str = "notifications.meetingSound";
/// Say so when a call starts or stops being recorded without asking.
const AUTO_RECORD_KEY: &str = "notifications.autoRecord";

/// Notification text in the app's language. `{source}` is replaced with the call app's name.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NotificationLabels {
    pub meeting_title: String,
    pub meeting_body: String,
    pub record: String,
    pub dismiss: String,
    pub recording_started: String,
    pub recording_stopped: String,
}

#[derive(Default)]
pub struct Labels(Mutex<NotificationLabels>);

#[tauri::command]
pub fn set_notification_labels(app: AppHandle, labels: NotificationLabels) {
    if app.try_state::<Labels>().is_none() {
        app.manage(Labels::default());
    }
    if let Ok(mut current) = app.state::<Labels>().0.lock() {
        *current = labels;
    }
}

pub fn labels(app: &AppHandle) -> NotificationLabels {
    app.try_state::<Labels>()
        .and_then(|labels| labels.0.lock().ok().map(|labels| labels.clone()))
        .unwrap_or_default()
}

fn setting(app: &AppHandle, key: &str) -> Option<serde_json::Value> {
    app.store(STORE_FILENAME).ok().and_then(|store| store.get(key))
}

/// Windows notifications carry the Record button; elsewhere the popup is the only style.
pub fn meeting_uses_system(app: &AppHandle) -> bool {
    cfg!(windows) && setting(app, MEETING_STYLE_KEY).as_ref().and_then(|value| value.as_str()) == Some("system")
}

pub fn meeting_seconds(app: &AppHandle) -> u64 {
    setting(app, MEETING_SECONDS_KEY)
        .and_then(|value| value.as_u64())
        .unwrap_or(0)
}

fn meeting_sound(app: &AppHandle) -> bool {
    setting(app, MEETING_SOUND_KEY)
        .and_then(|value| value.as_bool())
        .unwrap_or(true)
}

pub fn source_name(source: Source) -> &'static str {
    match source {
        Source::Zoom => "Zoom",
        Source::Teams => "Microsoft Teams",
        Source::Slack => "Slack",
        Source::Meet => "Google Meet",
    }
}

/// A plain system notification about a call recording that started or stopped on its own.
pub fn auto_record_notice(app: &AppHandle, started: bool, source: Option<Source>) {
    if setting(app, AUTO_RECORD_KEY).and_then(|value| value.as_bool()) == Some(false) {
        return;
    }
    let labels = labels(app);
    let text = if started {
        labels.recording_started
    } else {
        labels.recording_stopped
    };
    let text = text.replace("{source}", source.map(source_name).unwrap_or_default());
    if text.is_empty() {
        return;
    }
    app.notification().builder().title("Vibe").body(text).show().log_error();
}

/// A Windows notification with Record and Dismiss buttons. Windows only lets a notification stay
/// about 7 or 25 seconds, or until dismissed, so the chosen time picks the nearest of those.
/// `on_action` gets `"record"`, `"dismiss"`, or `None` for a click on the notification itself.
#[cfg(windows)]
pub fn show_meeting_toast(app: &AppHandle, source: Source, on_action: impl Fn(&AppHandle, Option<String>) + Send + 'static) {
    use tauri_winrt_notification::{Duration, Scenario, Sound, Toast};

    let labels = labels(app);
    let seconds = meeting_seconds(app);
    let mut toast = Toast::new(&app.config().identifier)
        .title(&labels.meeting_title.replace("{source}", source_name(source)))
        .text1(&labels.meeting_body)
        .add_button(&labels.record, "record")
        .add_button(&labels.dismiss, "dismiss")
        .sound(meeting_sound(app).then_some(Sound::Default));
    toast = match seconds {
        0 => toast.scenario(Scenario::Reminder),
        1..=7 => toast.duration(Duration::Short),
        _ => toast.duration(Duration::Long),
    };
    let handle = app.clone();
    if let Err(error) = toast
        .on_activated(move |action| {
            on_action(&handle, action);
            Ok(())
        })
        .show()
    {
        tracing::error!("could not show the meeting notification: {error}");
    }
}
