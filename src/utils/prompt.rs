//! Prompt formatting — the seam between the chat API's role/content language
//! and the flat text the tokenizer consumes.
//!
//! Roles ("system"/"user"/"assistant") are an *API-level* concept. The
//! inference engine only ever sees token ids; it has no notion of turns. This
//! module serializes structured turns into one prompt string, which the engine
//! handle then tokenizes.
//!
//! Today it uses a simple `Role: content` convention with a trailing
//! `Assistant:` cue. When a real model is wired in, this is the single place to
//! swap in that model's chat template (e.g. `<|im_start|>` markers) — nothing
//! below the tokenizer changes.

/// Format conversation turns into a single prompt string.
///
/// Each turn is `(role, content)`. The result labels every turn as
/// `Role: content` (one per line) and ends with an empty `Assistant:` cue so
/// the model continues as the assistant.
pub fn format_conversation<'a, I>(turns: I) -> String
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let mut prompt = String::new();
    for (role, content) in turns {
        prompt.push_str(&format!("{}: {}\n", display_role(role), content));
    }
    prompt.push_str("Assistant:");
    prompt
}

/// Capitalize a role label for display (`user` → `User`), defaulting unknown
/// roles to `Assistant` to match prior behavior.
fn display_role(role: &str) -> &'static str {
    match role {
        "system" => "System",
        "user" => "User",
        _ => "Assistant",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_turns_with_assistant_cue() {
        let prompt = format_conversation([
            ("system", "Be helpful."),
            ("user", "Hello"),
        ]);
        assert_eq!(prompt, "System: Be helpful.\nUser: Hello\nAssistant:");
    }

    #[test]
    fn unknown_role_defaults_to_assistant() {
        let prompt = format_conversation([("tool", "result")]);
        assert_eq!(prompt, "Assistant: result\nAssistant:");
    }

    #[test]
    fn empty_conversation_is_just_the_cue() {
        let prompt = format_conversation([]);
        assert_eq!(prompt, "Assistant:");
    }
}
