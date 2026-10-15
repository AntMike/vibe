use std::time::Duration;

fn main() {
    println!("mic owners: {:?}", meeting_detect::mic_owners());
    println!("{}", serde_json::to_string(&meeting_detect::detect()).unwrap());
    for state in meeting_detect::watch(Duration::from_secs(2)) {
        println!("{}", serde_json::to_string(&state).unwrap());
    }
}
