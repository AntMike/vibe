//! Windows-only: record just the call app (Teams or Slack) instead of every sound the PC
//! plays, and keep the mic track silent while the call app says you're muted.
//!
//! - Call audio: WASAPI process loopback on the app's process tree.
//! - Teams mute: the local "third-party app API" websocket (Teams settings → Privacy →
//!   Manage API → on). Pair once, in a solo "Meet now", by clicking Allow in Teams.
//! - Slack mute: UI Automation, reading the huddle mic button's label.
//!
//! Unknown mute state = record the mic. Losing your words is worse than keeping a muted aside.

use std::collections::VecDeque;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use eyre::{Context, Result};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};

/// Read by the mic stream callback on every buffer.
pub static MIC_MUTED: AtomicBool = AtomicBool::new(false);

const RATE: u32 = 48_000;
const CHANNELS: u16 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallApp {
    Teams,
    Slack,
}

impl CallApp {
    fn exe(self) -> &'static str {
        match self {
            CallApp::Teams => "ms-teams.exe",
            CallApp::Slack => "slack.exe",
        }
    }

    fn from_exe(name: &OsStr) -> Option<Self> {
        [CallApp::Teams, CallApp::Slack]
            .into_iter()
            .find(|app| name.eq_ignore_ascii_case(app.exe()))
    }
}

/// Walk up to the topmost process with the same exe. Capturing its tree covers the
/// renderers, WebView2 and audio service children.
fn root_of(system: &System, pid: Pid) -> Option<u32> {
    let mut process = system.process(pid)?;
    while let Some(parent) = process.parent().and_then(|pp| system.process(pp)) {
        if !parent.name().eq_ignore_ascii_case(process.name()) {
            break;
        }
        process = parent;
    }
    Some(process.pid().as_u32())
}

/// Whether a process image is the mic owner the OS reported: an exe path, or a Store
/// package family, which installs under `WindowsApps\<name>_<version>_<arch>__<publisher>\`.
fn owns(exe: &str, owner: &str) -> bool {
    let (exe, owner) = (exe.to_lowercase(), owner.to_lowercase());
    match owner.split_once('_') {
        Some((name, publisher)) if !owner.contains('\\') => {
            exe.contains(&format!("\\windowsapps\\{name}_")) && exe.contains(&format!("__{publisher}\\"))
        }
        _ => exe == owner,
    }
}

/// Pick the app to record: whoever holds the mic right now (Teams, Slack, a browser…; never
/// Vibe itself), Teams/Slack first so their mute state is followed. Nothing on the mic yet →
/// a running Teams/Slack. `None` → record all system audio as before.
pub fn find_call_app() -> Option<(Option<CallApp>, u32)> {
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::All,
        true,
        ProcessRefreshKind::nothing().with_exe(UpdateKind::OnlyIfNotSet),
    );
    let owners = meeting_detect::mic_owners();
    tracing::debug!("mic owners: {owners:?}");
    let on_mic: Vec<u32> = owners
        .iter()
        .filter_map(|owner| {
            let process = system
                .processes()
                .values()
                .find(|p| p.exe().is_some_and(|exe| owns(&exe.to_string_lossy(), owner)))?;
            root_of(&system, process.pid())
        })
        .filter(|pid| *pid != std::process::id())
        .collect();
    let app_of = |pid: u32| system.process(Pid::from_u32(pid)).and_then(|p| CallApp::from_exe(p.name()));
    let running_call_app = || {
        system
            .processes()
            .values()
            .find(|p| CallApp::from_exe(p.name()).is_some())
            .and_then(|p| root_of(&system, p.pid()))
    };
    let pid = on_mic
        .iter()
        .copied()
        .find(|pid| app_of(*pid).is_some())
        .or(on_mic.first().copied())
        .or_else(running_call_app)?;
    Some((app_of(pid), pid))
}

pub fn wav_spec() -> hound::WavSpec {
    hound::WavSpec {
        channels: CHANNELS,
        sample_rate: RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    }
}

/// Background work that runs until `stop()`.
pub struct Worker {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn spawn(name: &str, body: impl FnOnce(Arc<AtomicBool>) + Send + 'static) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let thread = thread::Builder::new().name(name.into()).spawn(move || body(flag)).ok();
        Self { stop, thread }
    }

    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().ok();
        }
    }
}

/// Capture the app's process tree as 48 kHz stereo f32. `on_samples` gets interleaved samples.
///
/// Process loopback may deliver nothing while the app is silent, so gaps are padded with
/// silence against the wall clock — the track has to stay in step with the mic for the merge.
pub fn start_app_loopback(pid: u32, mut on_samples: impl FnMut(&[f32]) + Send + 'static) -> Worker {
    Worker::spawn("call-loopback", move |stop| {
        if let Err(error) = loopback_loop(pid, &stop, &mut on_samples) {
            tracing::error!("call app capture failed: {error:#}");
        }
    })
}

fn loopback_loop(pid: u32, stop: &AtomicBool, on_samples: &mut impl FnMut(&[f32])) -> Result<()> {
    use wasapi::{initialize_mta, AudioClient, Direction, SampleType, StreamMode, WaveFormat};

    initialize_mta().ok().context("COM init")?;
    let format = WaveFormat::new(32, 32, &SampleType::Float, RATE as usize, CHANNELS as usize, None);
    let mut client = AudioClient::new_application_loopback_client(pid, true).context("process loopback")?;
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: 0,
    };
    client.initialize_client(&format, &Direction::Capture, &mode)?;
    let event = client.set_get_eventhandle()?;
    let capture = client.get_audiocaptureclient()?;
    client.start_stream()?;
    tracing::info!("capturing call app audio from pid {pid}");

    let frame_bytes = 4 * CHANNELS as usize;
    let started = Instant::now();
    let mut frames_written: u64 = 0;
    let mut bytes = VecDeque::new();
    let mut samples = Vec::new();
    let silence = vec![0.0f32; RATE as usize / 10 * CHANNELS as usize];

    while !stop.load(Ordering::Relaxed) {
        if event.wait_for_event(100).is_err() {
            // Nothing played for 100 ms: fill up to the wall clock.
            let expected = (started.elapsed().as_secs_f64() * RATE as f64) as u64;
            let mut missing = expected.saturating_sub(frames_written);
            while missing > 0 {
                let frames = missing.min((silence.len() / CHANNELS as usize) as u64);
                on_samples(&silence[..frames as usize * CHANNELS as usize]);
                frames_written += frames;
                missing -= frames;
            }
            continue;
        }
        while capture.get_next_packet_size()?.unwrap_or(0) > 0 {
            let before = bytes.len();
            let info = capture.read_from_device_to_deque(&mut bytes)?;
            if info.flags.silent {
                bytes.iter_mut().skip(before).for_each(|b| *b = 0);
            }
        }
        let whole = bytes.len() / frame_bytes * frame_bytes;
        samples.clear();
        samples.extend(
            bytes
                .drain(..whole)
                .collect::<Vec<u8>>()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b)),
        );
        frames_written += (samples.len() / CHANNELS as usize) as u64;
        on_samples(&samples);
    }
    client.stop_stream().ok();
    Ok(())
}

/// Keep `MIC_MUTED` in sync with the call app until stopped. Resets to unmuted on stop.
pub fn start_mute_watch(app: CallApp, pid: u32, teams_token_file: PathBuf) -> Worker {
    MIC_MUTED.store(false, Ordering::Relaxed);
    Worker::spawn("call-mute-watch", move |stop| {
        match app {
            CallApp::Teams => teams::watch(&stop, &teams_token_file),
            CallApp::Slack => slack::watch(&stop, pid),
        }
        MIC_MUTED.store(false, Ordering::Relaxed);
    })
}

mod teams {
    use super::MIC_MUTED;
    use std::net::TcpStream;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;
    use tungstenite::{stream::MaybeTlsStream, Message, WebSocket};

    pub fn watch(stop: &AtomicBool, token_file: &Path) {
        while !stop.load(Ordering::Relaxed) {
            if let Err(error) = session(stop, token_file) {
                tracing::debug!("teams api: {error}");
            }
            // Teams closed, API off, or connection dropped: state unknown → record the mic.
            MIC_MUTED.store(false, Ordering::Relaxed);
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    fn session(stop: &AtomicBool, token_file: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let saved = std::fs::read_to_string(token_file).unwrap_or_default();
        let paired = !saved.trim().is_empty();
        // Unpaired clients still need some token in the URL; Teams swaps it for a real one on pairing.
        let token = if paired {
            saved.trim().to_string()
        } else {
            uuid::Uuid::new_v4().to_string()
        };
        let url = format!(
            "ws://127.0.0.1:8124?token={}&protocol-version=2.0.0&manufacturer=Vibe&device=Vibe&app=Vibe&app-version=1.0.0",
            token
        );
        let (mut socket, _) = tungstenite::connect(url)?;
        if let MaybeTlsStream::Plain(tcp) = socket.get_ref() {
            tcp.set_read_timeout(Some(Duration::from_millis(500)))?;
        }
        let mut pair_requested = false;
        while !stop.load(Ordering::Relaxed) {
            let text = match socket.read() {
                Ok(Message::Text(text)) => text,
                Ok(Message::Close(_)) => return Ok(()),
                Ok(_) => continue,
                Err(tungstenite::Error::Io(e))
                    if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) =>
                {
                    continue
                }
                Err(e) => return Err(e.into()),
            };
            let msg: serde_json::Value = serde_json::from_str(&text)?;
            if let Some(token) = msg["tokenRefresh"].as_str() {
                std::fs::write(token_file, token)?;
                tracing::info!("paired with Teams");
            }
            if let Some(muted) = msg["meetingUpdate"]["meetingState"]["isMuted"].as_bool() {
                MIC_MUTED.store(muted, Ordering::Relaxed);
            }
            let can_pair = msg["meetingUpdate"]["meetingPermissions"]["canPair"].as_bool() == Some(true);
            if can_pair && !paired && !pair_requested {
                // Any command triggers Teams' "Allow" prompt; a 👍 is the least intrusive one.
                request_pairing(&mut socket)?;
                pair_requested = true;
            }
        }
        socket.close(None).ok();
        Ok(())
    }

    fn request_pairing(socket: &mut WebSocket<MaybeTlsStream<TcpStream>>) -> tungstenite::Result<()> {
        let body = r#"{"action":"send-reaction","parameters":{"type":"like"},"requestId":1}"#;
        socket.send(Message::text(body))
    }
}

mod slack {
    use super::MIC_MUTED;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;
    use windows::core::BSTR;
    use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED};
    use windows::Win32::System::Variant::VARIANT;
    use windows::Win32::UI::Accessibility::{
        CUIAutomation, IUIAutomation, IUIAutomationElement, TreeScope_Children, TreeScope_Descendants, UIA_ButtonControlTypeId,
        UIA_ControlTypePropertyId, UIA_ProcessIdPropertyId,
    };

    /// `Some(true)` muted, `Some(false)` live, `None` not a huddle mic button.
    // ponytail: label heuristic; calibrate against the names logged on first scan if Slack renames it.
    pub(super) fn mute_from_label(label: &str) -> Option<bool> {
        let label = label.trim().to_lowercase();
        let about_mic = label.contains("mic") || label == "mute" || label == "unmute";
        if !about_mic {
            return None;
        }
        if label.starts_with("unmute") {
            Some(true)
        } else if label.starts_with("mute") {
            Some(false)
        } else {
            None
        }
    }

    pub fn watch(stop: &AtomicBool, pid: u32) {
        if let Err(error) = run(stop, pid) {
            tracing::error!("slack mute watch failed: {error}");
        }
    }

    fn run(stop: &AtomicBool, pid: u32) -> windows::core::Result<()> {
        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED).ok()? };
        let uia: IUIAutomation = unsafe { CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER)? };
        let mut button: Option<IUIAutomationElement> = None;
        let mut logged = false;
        while !stop.load(Ordering::Relaxed) {
            let state = button
                .as_ref()
                .and_then(|b| unsafe { b.CurrentName() }.ok())
                .and_then(|name| mute_from_label(&name.to_string()));
            match state {
                Some(muted) => MIC_MUTED.store(muted, Ordering::Relaxed),
                None => {
                    // Not in a huddle, or the button was re-rendered: search again.
                    button = find_mic_button(&uia, pid, !logged).unwrap_or(None);
                    logged = true;
                    if button.is_none() {
                        MIC_MUTED.store(false, Ordering::Relaxed);
                        std::thread::sleep(Duration::from_secs(1));
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(300));
        }
        Ok(())
    }

    fn find_mic_button(uia: &IUIAutomation, pid: u32, log_names: bool) -> windows::core::Result<Option<IUIAutomationElement>> {
        unsafe {
            let root = uia.GetRootElement()?;
            let by_pid = uia.CreatePropertyCondition(UIA_ProcessIdPropertyId, &VARIANT::from(pid as i32))?;
            let by_button = uia.CreatePropertyCondition(UIA_ControlTypePropertyId, &VARIANT::from(UIA_ButtonControlTypeId.0))?;
            let windows = root.FindAll(TreeScope_Children, &by_pid)?;
            for w in 0..windows.Length()? {
                let buttons = windows.GetElement(w)?.FindAll(TreeScope_Descendants, &by_button)?;
                for b in 0..buttons.Length()? {
                    let element = buttons.GetElement(b)?;
                    let name: BSTR = element.CurrentName().unwrap_or_default();
                    let name = name.to_string();
                    if log_names && name.to_lowercase().contains("mute") {
                        tracing::info!("slack button: {name:?}");
                    }
                    if mute_from_label(&name).is_some() {
                        return Ok(Some(element));
                    }
                }
            }
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::owns;
    use super::slack::mute_from_label;

    #[test]
    fn mic_owner_matching() {
        let teams = r"C:\Program Files\WindowsApps\MSTeams_25198.1112.3855.2071_x64__8wekyb3d8bbwe\ms-teams.exe";
        assert!(owns(teams, "MSTeams_8wekyb3d8bbwe"));
        assert!(!owns(teams, "com.tinyspeck.slackdesktop_8yrtsj140pw4g"));
        let chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe";
        assert!(owns(chrome, r"C:\Program Files\Google\Chrome\Application\chrome.exe"));
        assert!(!owns(chrome, r"C:\Program Files\Mozilla Firefox\firefox.exe"));
    }

    #[test]
    fn slack_labels() {
        assert_eq!(mute_from_label("Mute mic"), Some(false));
        assert_eq!(mute_from_label("Unmute mic"), Some(true));
        assert_eq!(mute_from_label("Unmute"), Some(true));
        assert_eq!(mute_from_label("Mute channel"), None);
        assert_eq!(mute_from_label("Microphone settings"), None);
    }
}
