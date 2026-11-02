use sha2::{Digest, Sha256};

#[allow(dead_code)]
pub fn verify_hash_chain(events: &[(&str, &str, &str)]) -> (bool, Option<usize>) {
    // events: slice of (event_id, payload_str, previous_hash)
    // Returns (valid, failing_index)
    let mut expected_prev = String::from("0000000000000000000000000000000000000000000000000000000000000000");

    for (idx, (event_id, payload, recorded_prev)) in events.iter().enumerate() {
        if *recorded_prev != expected_prev {
            return (false, Some(idx));
        }

        let mut hasher = Sha256::new();
        hasher.update(recorded_prev.as_bytes());
        hasher.update(b":");
        hasher.update(event_id.as_bytes());
        hasher.update(b":");
        hasher.update(payload.as_bytes());
        let digest = hasher.finalize();
        expected_prev = hex::encode(digest);
    }

    (true, None)
}
