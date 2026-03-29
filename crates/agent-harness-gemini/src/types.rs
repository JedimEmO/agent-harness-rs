#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// --- Request types ---

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiRequest {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<GeminiToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GeminiPart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponse,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    pub mime_type: String,
    pub data: String, // base64-encoded
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default = "default_empty_object")]
    pub args: serde_json::Value,
}

fn default_empty_object() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiToolConfig {
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

// --- Response types ---

#[derive(Debug, Deserialize)]
pub struct GeminiResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: Option<GeminiContent>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    #[serde(default)]
    pub prompt_token_count: u32,
    #[serde(default)]
    pub candidates_token_count: u32,
    #[serde(default)]
    pub total_token_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_part_text_serde() {
        let part = GeminiPart::Text {
            text: "hi".to_string(),
        };
        let json = serde_json::to_string(&part).unwrap();
        assert_eq!(json, r#"{"text":"hi"}"#);

        let deserialized: GeminiPart = serde_json::from_str(&json).unwrap();
        match deserialized {
            GeminiPart::Text { text } => assert_eq!(text, "hi"),
            _ => panic!("expected Text variant"),
        }
    }

    #[test]
    fn gemini_part_function_call_serde() {
        let part = GeminiPart::FunctionCall {
            function_call: FunctionCall {
                name: "get_weather".to_string(),
                args: serde_json::json!({"city": "London"}),
            },
        };
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("functionCall"));
        assert!(json.contains("get_weather"));

        let deserialized: GeminiPart = serde_json::from_str(&json).unwrap();
        match deserialized {
            GeminiPart::FunctionCall { function_call } => {
                assert_eq!(function_call.name, "get_weather");
                assert_eq!(function_call.args["city"], "London");
            }
            _ => panic!("expected FunctionCall variant"),
        }
    }

    #[test]
    fn gemini_part_inline_data_serde() {
        let part = GeminiPart::InlineData {
            inline_data: InlineData {
                mime_type: "image/png".to_string(),
                data: "aGVsbG8=".to_string(),
            },
        };
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("inlineData"));
        assert!(json.contains("mimeType"));

        let deserialized: GeminiPart = serde_json::from_str(&json).unwrap();
        match deserialized {
            GeminiPart::InlineData { inline_data } => {
                assert_eq!(inline_data.mime_type, "image/png");
                assert_eq!(inline_data.data, "aGVsbG8=");
            }
            _ => panic!("expected InlineData variant"),
        }
    }

    #[test]
    fn gemini_response_deserialize_text() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello, world!"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        }"#;
        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.candidates.len(), 1);
        let content = response.candidates[0].content.as_ref().unwrap();
        assert_eq!(content.role.as_deref(), Some("model"));
        assert_eq!(content.parts.len(), 1);
        match &content.parts[0] {
            GeminiPart::Text { text } => assert_eq!(text, "Hello, world!"),
            _ => panic!("expected Text part"),
        }
        let usage = response.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, 10);
        assert_eq!(usage.candidates_token_count, 5);
        assert_eq!(usage.total_token_count, 15);
    }

    #[test]
    fn gemini_response_deserialize_function_calls() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}]
                },
                "finishReason": "STOP"
            }]
        }"#;
        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        let content = response.candidates[0].content.as_ref().unwrap();
        match &content.parts[0] {
            GeminiPart::FunctionCall { function_call } => {
                assert_eq!(function_call.name, "get_weather");
                assert_eq!(function_call.args["city"], "NYC");
            }
            _ => panic!("expected FunctionCall part"),
        }
    }

    #[test]
    fn gemini_response_empty_candidates() {
        let json = r#"{"candidates": []}"#;
        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        assert!(response.candidates.is_empty());
        assert!(response.usage_metadata.is_none());
    }

    // =========================================================================
    // ADVERSARIAL TESTS
    // =========================================================================

    #[test]
    fn gemini_part_untagged_ambiguity_text_wins_over_function_response() {
        // BUG PROBE: With #[serde(untagged)], serde tries variants in declaration order.
        // A JSON object with both "text" AND "functionResponse" fields.
        // Since Text is declared first, if the JSON has a "text" field, it will match Text
        // and ignore the functionResponse.
        let json = r#"{"text": "hello", "functionResponse": {"name": "test", "response": {}}}"#;
        let part: GeminiPart = serde_json::from_str(json).unwrap();
        match part {
            GeminiPart::Text { text } => {
                assert_eq!(text, "hello");
                // The functionResponse is silently lost
            }
            _ => panic!("Expected Text variant to win due to declaration order in untagged enum"),
        }
    }

    #[test]
    fn gemini_part_untagged_empty_object_fails() {
        // BUG PROBE: An empty JSON object {} should fail to deserialize as any variant.
        // Text requires "text" field, InlineData requires "inlineData", etc.
        let result = serde_json::from_str::<GeminiPart>("{}");
        assert!(result.is_err(), "empty object should not match any GeminiPart variant");
    }

    #[test]
    fn gemini_part_function_call_with_empty_args() {
        // BUG PROBE: functionCall with no args field. FunctionCall.args has #[serde(default)]
        // so it should default to Value::Null, not Value::Object(empty).
        let json = r#"{"functionCall": {"name": "do_thing"}}"#;
        let part: GeminiPart = serde_json::from_str(json).unwrap();
        match part {
            GeminiPart::FunctionCall { function_call } => {
                assert_eq!(function_call.name, "do_thing");
                // Missing args should default to empty object
                assert_eq!(
                    function_call.args,
                    serde_json::Value::Object(serde_json::Map::new()),
                    "missing args should default to empty object"
                );
            }
            _ => panic!("expected FunctionCall variant"),
        }
    }

    #[test]
    fn gemini_part_text_and_function_call_in_same_parts_array() {
        // BUG PROBE: A parts array containing both text and function call.
        // This is valid Gemini API behavior. The parts should deserialize independently.
        let json = r#"{
            "role": "model",
            "parts": [
                {"text": "Let me check the weather"},
                {"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}
            ]
        }"#;
        let content: GeminiContent = serde_json::from_str(json).unwrap();
        assert_eq!(content.parts.len(), 2);
        assert!(matches!(&content.parts[0], GeminiPart::Text { text } if text == "Let me check the weather"));
        assert!(matches!(&content.parts[1], GeminiPart::FunctionCall { .. }));
    }

    #[test]
    fn gemini_response_null_content_in_candidate() {
        // BUG PROBE: Candidate with null content (happens on safety blocks)
        let json = r#"{
            "candidates": [{
                "content": null,
                "finishReason": "SAFETY"
            }]
        }"#;
        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        assert!(response.candidates[0].content.is_none());
    }

    #[test]
    fn gemini_response_missing_content_field() {
        // BUG PROBE: Candidate without content field at all
        let json = r#"{
            "candidates": [{
                "finishReason": "SAFETY"
            }]
        }"#;
        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        assert!(response.candidates[0].content.is_none());
    }

    #[test]
    fn gemini_response_usage_metadata_zero_defaults() {
        // BUG PROBE: usage_metadata with missing token counts should default to 0
        let json = r#"{
            "candidates": [],
            "usageMetadata": {}
        }"#;
        let response: GeminiResponse = serde_json::from_str(json).unwrap();
        let usage = response.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, 0);
        assert_eq!(usage.candidates_token_count, 0);
        assert_eq!(usage.total_token_count, 0);
    }

    #[test]
    fn gemini_part_inline_data_could_shadow_function_call() {
        // BUG PROBE: JSON with both "inlineData" and "functionCall".
        // InlineData is declared before FunctionCall, so it wins.
        let json = r#"{
            "inlineData": {"mimeType": "image/png", "data": "abc="},
            "functionCall": {"name": "test", "args": {}}
        }"#;
        let part: GeminiPart = serde_json::from_str(json).unwrap();
        // InlineData wins because it's declared second (after Text fails because no "text" field)
        assert!(
            matches!(part, GeminiPart::InlineData { .. }),
            "InlineData should win over FunctionCall when both fields present"
        );
    }
}
