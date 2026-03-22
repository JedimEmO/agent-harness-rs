diesel::table! {
    agent_sessions (id) {
        id -> Text,
        scope_id -> Text,
        title -> Nullable<Text>,
        created_at -> Text,
        updated_at -> Text,
    }
}

diesel::table! {
    agent_messages (id) {
        id -> Text,
        session_id -> Text,
        role -> Text,
        content_type -> Text,
        content -> Text,
        created_at -> Text,
    }
}

diesel::table! {
    agent_memories (id) {
        id -> Text,
        scope_id -> Text,
        key -> Text,
        content -> Text,
        category -> Text,
        created_at -> Text,
        updated_at -> Text,
    }
}
