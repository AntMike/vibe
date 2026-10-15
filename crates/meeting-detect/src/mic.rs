use crate::MicUsage;

/// Return the strongest microphone-ownership signal available on this OS.
pub(crate) fn current_usage() -> MicUsage {
    platform::current_usage()
}

#[cfg(any(target_os = "linux", test))]
fn process_from_properties(properties: &serde_json::Map<String, serde_json::Value>) -> crate::ProcessInfo {
    let string = |key: &str| properties.get(key).and_then(serde_json::Value::as_str);
    let executable = string("application.process.binary").map(str::to_string);
    let name = string("application.name")
        .or_else(|| string("media.name"))
        .or(executable.as_deref())
        .unwrap_or("microphone client")
        .to_string();
    let pid = properties.get("application.process.id").and_then(|value| {
        value
            .as_u64()
            .and_then(|pid| u32::try_from(pid).ok())
            .or_else(|| value.as_str().and_then(|pid| pid.parse().ok()))
    });
    crate::ProcessInfo {
        pid,
        name,
        executable,
        bundle_id: None,
    }
}

#[cfg(any(target_os = "linux", test))]
fn parse_pactl_json(input: &str) -> Option<MicUsage> {
    let streams = serde_json::from_str::<serde_json::Value>(input).ok()?.as_array()?.clone();
    let mut processes = Vec::new();
    let mut active = false;
    for stream in streams {
        let Some(stream) = stream.as_object() else { continue };
        let is_active = match stream.get("corked").and_then(serde_json::Value::as_bool) {
            Some(corked) => !corked,
            None => stream
                .get("state")
                .and_then(serde_json::Value::as_str)
                .map(|state| state.eq_ignore_ascii_case("running"))
                .unwrap_or(true),
        };
        if !is_active {
            continue;
        }
        active = true;
        if let Some(properties) = stream.get("properties").and_then(serde_json::Value::as_object) {
            let process = process_from_properties(properties);
            if !processes.contains(&process) {
                processes.push(process);
            }
        }
    }
    Some(MicUsage { active, processes })
}

#[cfg(any(target_os = "linux", test))]
fn unquote_pactl_value(value: &str) -> String {
    let value = value.trim();
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(value)
        .replace("\\\"", "\"")
        .replace("\\\\", "\\")
}

#[cfg(any(target_os = "linux", test))]
fn parse_pactl_text(input: &str) -> Option<MicUsage> {
    let mut saw_stream = false;
    let mut streams: Vec<(bool, serde_json::Map<String, serde_json::Value>)> = Vec::new();
    let mut current: Option<(bool, serde_json::Map<String, serde_json::Value>)> = None;

    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Source Output #") {
            saw_stream = true;
            if let Some(stream) = current.take() {
                streams.push(stream);
            }
            current = Some((true, serde_json::Map::new()));
            continue;
        }
        let Some((active, properties)) = current.as_mut() else {
            continue;
        };
        if let Some(value) = trimmed.strip_prefix("Corked:") {
            *active = !value.trim().eq_ignore_ascii_case("yes");
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if matches!(
            key,
            "application.name" | "application.process.binary" | "application.process.id" | "media.name"
        ) {
            properties.insert(key.to_string(), serde_json::Value::String(unquote_pactl_value(value)));
        }
    }
    if let Some(stream) = current {
        streams.push(stream);
    }
    if !saw_stream {
        return None;
    }

    let mut processes = Vec::new();
    let mut active = false;
    for (stream_active, properties) in streams {
        if !stream_active {
            continue;
        }
        active = true;
        let process = process_from_properties(&properties);
        if !processes.contains(&process) {
            processes.push(process);
        }
    }
    Some(MicUsage { active, processes })
}

#[cfg(any(target_os = "windows", test))]
fn decode_windows_executable_key(key: &str) -> String {
    key.replace('#', "\\")
}

#[cfg(target_os = "windows")]
mod platform {
    use crate::{MicUsage, ProcessInfo};
    use std::path::Path;
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
    use winreg::{enums::HKEY_CURRENT_USER, RegKey};

    const CONSENT_STORE: &str = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone";

    fn is_active(key: &RegKey) -> bool {
        matches!(key.get_value::<u64, _>("LastUsedTimeStop"), Ok(0))
    }

    fn nonpackaged_owner(owner_key: &str) -> ProcessInfo {
        let executable = super::decode_windows_executable_key(owner_key);
        let name = Path::new(&executable)
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .unwrap_or(owner_key)
            .to_string();
        ProcessInfo {
            pid: None,
            name,
            executable: Some(executable),
            bundle_id: None,
        }
    }

    fn collect_nonpackaged(key: &RegKey, owner: Option<&str>, processes: &mut Vec<ProcessInfo>) {
        if is_active(key) {
            if let Some(owner) = owner {
                let process = nonpackaged_owner(owner);
                if !processes.contains(&process) {
                    processes.push(process);
                }
            }
        }
        for child_name in key.enum_keys().flatten() {
            if let Ok(child) = key.open_subkey(&child_name) {
                collect_nonpackaged(&child, owner.or(Some(&child_name)), processes);
            }
        }
    }

    /// Resolve pids by matching image paths against the process list.
    ///
    /// Enumerating every process is by far the most expensive thing this module does, so it is
    /// confined to the one case that needs the result: matching a browser window title to find
    /// Meet. Zoom and Teams are identified by the consent-store key alone, and an idle machine
    /// never pays for it at all.
    fn attach_pids(processes: &mut [ProcessInfo]) {
        let mut wanted: Vec<&mut ProcessInfo> = processes
            .iter_mut()
            .filter(|owner| owner.executable.is_some() && crate::process::source(owner) == Some(crate::ProcessKind::Browser))
            .collect();
        if wanted.is_empty() {
            return;
        }

        let mut system = System::new();
        system.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing().with_exe(UpdateKind::OnlyIfNotSet),
        );
        for owner in &mut wanted {
            let Some(expected) = owner.executable.as_deref() else {
                continue;
            };
            owner.pid = system.processes().values().find_map(|process| {
                let actual = process.exe()?.to_string_lossy();
                actual.eq_ignore_ascii_case(expected).then(|| process.pid().as_u32())
            });
        }
    }

    /// Processes with a running capture stream on any microphone, straight from the audio engine.
    fn capture_session_pids() -> windows::core::Result<Vec<u32>> {
        use windows::core::Interface;
        use windows::Win32::Foundation::S_OK;
        use windows::Win32::Media::Audio::{
            eCapture, AudioSessionStateActive, IAudioSessionControl2, IAudioSessionManager2, IMMDeviceEnumerator,
            MMDeviceEnumerator, DEVICE_STATE_ACTIVE,
        };
        use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED};

        thread_local! {
            // Once per thread. A thread that is already STA reports RPC_E_CHANGED_MODE, and COM works there too.
            static COM: () = unsafe { _ = CoInitializeEx(None, COINIT_MULTITHREADED) };
        }
        COM.with(|_| ());
        let mut pids = Vec::new();
        unsafe {
            let devices: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
            let mics = devices.EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE)?;
            for m in 0..mics.GetCount()? {
                let manager: IAudioSessionManager2 = mics.Item(m)?.Activate(CLSCTX_ALL, None)?;
                let sessions = manager.GetSessionEnumerator()?;
                for s in 0..sessions.GetCount()? {
                    let session: IAudioSessionControl2 = sessions.GetSession(s)?.cast()?;
                    if session.GetState()? != AudioSessionStateActive || session.IsSystemSoundsSession() == S_OK {
                        continue;
                    }
                    let pid = session.GetProcessId()?;
                    if pid != 0 && !pids.contains(&pid) {
                        pids.push(pid);
                    }
                }
            }
        }
        Ok(pids)
    }

    /// Who owns each capture session. Electron apps capture from a child of the same exe, and
    /// WebView2 hosts from `msedgewebview2.exe`, so the first ancestor that is a meeting app wins.
    // ponytail: full process refresh on every poll while any mic is open; cache by pid if it shows in CPU.
    fn session_owners() -> Vec<ProcessInfo> {
        let pids = capture_session_pids().unwrap_or_else(|error| {
            tracing::debug!("capture sessions: {error}");
            Vec::new()
        });
        if pids.is_empty() {
            return Vec::new();
        }
        let mut system = System::new();
        system.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing().with_exe(UpdateKind::OnlyIfNotSet),
        );
        let info = |process: &sysinfo::Process| ProcessInfo {
            pid: None,
            name: process.name().to_string_lossy().into_owned(),
            executable: process.exe().map(|exe| exe.to_string_lossy().into_owned()),
            bundle_id: None,
        };
        pids.into_iter()
            .filter_map(|pid| {
                let own = system.process(Pid::from_u32(pid))?;
                let ancestors = std::iter::successors(Some(own), |p| p.parent().and_then(|parent| system.process(parent)));
                // Bounded: a dead parent's pid can be reused by a descendant, making the chain a loop.
                let owner = ancestors
                    .take(16)
                    .map(info)
                    .find(|owner| crate::process::source(owner).is_some())
                    .unwrap_or_else(|| info(own));
                Some(owner)
            })
            .collect()
    }

    pub(super) fn current_usage() -> MicUsage {
        let mut processes = Vec::new();
        // ponytail: since Windows update KB5127648 (build 26300.9711) the consent store stopped recording mic
        // use; kept for systems where it still does, the capture sessions below are what actually see a call.
        if let Ok(store) = RegKey::predef(HKEY_CURRENT_USER).open_subkey(CONSENT_STORE) {
            for child_name in store.enum_keys().flatten() {
                let Ok(child) = store.open_subkey(&child_name) else {
                    continue;
                };
                if child_name.eq_ignore_ascii_case("NonPackaged") {
                    collect_nonpackaged(&child, None, &mut processes);
                } else if is_active(&child) {
                    processes.push(ProcessInfo {
                        pid: None,
                        name: child_name.clone(),
                        executable: None,
                        bundle_id: Some(child_name),
                    });
                }
            }
        }
        for owner in session_owners() {
            if !processes.contains(&owner) {
                processes.push(owner);
            }
        }
        attach_pids(&mut processes);
        MicUsage {
            active: !processes.is_empty(),
            processes,
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use crate::MicUsage;
    use std::process::Command;

    fn output(args: &[&str]) -> Option<String> {
        let output = Command::new("pactl").args(args).output().ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
    }

    pub(super) fn current_usage() -> MicUsage {
        if let Some(usage) = output(&["-f", "json", "list", "source-outputs"])
            .as_deref()
            .and_then(super::parse_pactl_json)
        {
            return usage;
        }
        output(&["list", "source-outputs"])
            .as_deref()
            .and_then(super::parse_pactl_text)
            .unwrap_or_default()
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use crate::MicUsage;
    use coreaudio_sys::{
        kAudioDevicePropertyDeviceIsRunningSomewhere, kAudioHardwarePropertyDefaultInputDevice, kAudioObjectPropertyElementMain,
        kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject, AudioDeviceID, AudioObjectGetPropertyData,
        AudioObjectPropertyAddress,
    };
    use std::{ffi::c_void, mem::size_of, ptr};

    unsafe fn property<T: Copy>(object: u32, selector: u32, value: &mut T) -> bool {
        let address = AudioObjectPropertyAddress {
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain,
        };
        let mut size = size_of::<T>() as u32;
        unsafe {
            AudioObjectGetPropertyData(object, &address, 0, ptr::null(), &mut size, value as *mut T as *mut c_void) == 0
                && size as usize == size_of::<T>()
        }
    }

    pub(super) fn current_usage() -> MicUsage {
        let mut device: AudioDeviceID = 0;
        if !unsafe {
            property(
                kAudioObjectSystemObject,
                kAudioHardwarePropertyDefaultInputDevice,
                &mut device,
            )
        } || device == 0
        {
            return MicUsage::default();
        }
        let mut running: u32 = 0;
        let active = unsafe { property(device, kAudioDevicePropertyDeviceIsRunningSomewhere, &mut running) } && running != 0;
        MicUsage {
            active,
            processes: Vec::new(),
        }
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
mod platform {
    use crate::MicUsage;

    pub(super) fn current_usage() -> MicUsage {
        MicUsage::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_active_and_corked_pactl_json_streams() {
        let input = r#"[
          {"corked": false, "properties": {
            "application.name": "Firefox", "application.process.binary": "firefox", "application.process.id": "42"
          }},
          {"corked": true, "properties": {"application.name": "Ignored"}}
        ]"#;
        let usage = parse_pactl_json(input).unwrap();
        assert!(usage.active);
        assert_eq!(usage.processes.len(), 1);
        assert_eq!(usage.processes[0].pid, Some(42));
        assert_eq!(usage.processes[0].executable.as_deref(), Some("firefox"));
    }

    #[test]
    fn rejects_malformed_json_and_parses_text_fallback() {
        assert!(parse_pactl_json("not json").is_none());
        let text = r#"
Source Output #17
    Corked: no
    Properties:
        application.name = "Microsoft Teams"
        application.process.binary = "ms-teams"
        application.process.id = "501"
Source Output #18
    Corked: yes
    Properties:
        application.name = "Muted"
"#;
        let usage = parse_pactl_text(text).unwrap();
        assert!(usage.active);
        assert_eq!(usage.processes.len(), 1);
        assert_eq!(usage.processes[0].name, "Microsoft Teams");
        assert_eq!(usage.processes[0].pid, Some(501));
        assert!(parse_pactl_text("garbage").is_none());
    }

    #[test]
    fn decodes_windows_nonpackaged_executable_keys() {
        assert_eq!(
            decode_windows_executable_key(r"C:#Program Files#Zoom#bin#Zoom.exe"),
            r"C:\Program Files\Zoom\bin\Zoom.exe"
        );
    }
}
