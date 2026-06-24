use goose_providers::conversation::Conversation;
use rmcp::model::Role;

const ANCHOR_MARKER: &str = ">>>";

/// Render `conversation` into the `user:/assistant:` form used at training
/// time. The most recent user message is the anchor, marked with `>>>`;
/// everything before it is the context.
///
/// Returns `None` if no user message exists. We don't enforce a token budget
/// here — encoders truncate to their max sequence length, and the anchor is at
/// the end so truncation drops the oldest context first.
pub fn render_for_routing(conversation: &Conversation) -> Option<String> {
    let messages = conversation.messages();
    let anchor_idx = messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, m)| matches!(m.role, Role::User))
        .map(|(i, _)| i)?;

    let mut lines = Vec::with_capacity(anchor_idx + 1);
    for (i, msg) in messages.iter().take(anchor_idx + 1).enumerate() {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        let text = msg.as_concat_text();
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if i == anchor_idx {
            lines.push(format!("{} {}: {}", ANCHOR_MARKER, role, text));
        } else {
            lines.push(format!("{}: {}", role, text));
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goose_providers::conversation::message::Message;

    fn convo(msgs: Vec<Message>) -> Conversation {
        Conversation::new_unvalidated(msgs)
    }

    #[test]
    fn render_empty_returns_none() {
        let c = convo(vec![]);
        assert!(render_for_routing(&c).is_none());
    }

    #[test]
    fn render_single_user_message() {
        let c = convo(vec![Message::user().with_text("hi there")]);
        let r = render_for_routing(&c).expect("some");
        assert_eq!(r, ">>> user: hi there");
    }

    #[test]
    fn render_multi_turn_marks_last_user() {
        let c = convo(vec![
            Message::user().with_text("what is 2+2"),
            Message::assistant().with_text("4"),
            Message::user().with_text("now squared"),
        ]);
        let r = render_for_routing(&c).expect("some");
        assert_eq!(r, "user: what is 2+2\nassistant: 4\n>>> user: now squared");
    }

    #[test]
    fn render_skips_empty_messages() {
        let c = convo(vec![
            Message::user().with_text("real question"),
            Message::assistant().with_text(""),
            Message::user().with_text("follow up"),
        ]);
        let r = render_for_routing(&c).expect("some");
        assert_eq!(r, "user: real question\n>>> user: follow up");
    }
}
