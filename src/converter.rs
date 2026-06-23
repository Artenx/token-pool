//! API 格式转换器
//! 支持 OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 三种格式互相转换

use serde_json::Value;

/// 统一的中间表示
#[derive(Debug, Clone)]
pub struct UnifiedRequest {
    pub model: String,
    pub messages: Vec<UnifiedMessage>,
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub stream: bool,
    pub system: Option<String>,
}

#[derive(Debug, Clone)]
pub struct UnifiedMessage {
    pub role: String,
    pub content: String,
}

/// 从 OpenAI Chat Completions 格式解析
pub fn parse_openai(body: &Value) -> UnifiedRequest {
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let max_tokens = body.get("max_tokens").and_then(|v| v.as_u64());
    let temperature = body.get("temperature").and_then(|v| v.as_f64());
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let messages: Vec<UnifiedMessage> = body.get("messages")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter().filter_map(|msg| {
                let role = msg.get("role")?.as_str()?.to_string();
                let content = msg.get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Some(UnifiedMessage { role, content })
            }).collect()
        })
        .unwrap_or_default();

    // 提取 system message
    let system = messages.iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.clone());

    let messages = messages.into_iter()
        .filter(|m| m.role != "system")
        .collect();

    UnifiedRequest { model, messages, max_tokens, temperature, stream, system }
}

/// 从 OpenAI Responses 格式解析
pub fn parse_openai_responses(body: &Value) -> UnifiedRequest {
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let max_tokens = body.get("max_output_tokens").and_then(|v| v.as_u64());
    let temperature = body.get("temperature").and_then(|v| v.as_f64());
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    // input 可以是字符串或数组
    let messages = match body.get("input") {
        Some(Value::String(s)) => {
            vec![UnifiedMessage { role: "user".to_string(), content: s.clone() }]
        }
        Some(Value::Array(arr)) => {
            arr.iter().filter_map(|item| {
                let role = item.get("role")?.as_str()?.to_string();
                let content = item.get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Some(UnifiedMessage { role, content })
            }).collect()
        }
        _ => vec![],
    };

    // instructions 作为 system
    let system = body.get("instructions")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    UnifiedRequest { model, messages, max_tokens, temperature, stream, system }
}

/// 从 Anthropic Messages 格式解析
pub fn parse_anthropic(body: &Value) -> UnifiedRequest {
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let max_tokens = body.get("max_tokens").and_then(|v| v.as_u64());
    let temperature = body.get("temperature").and_then(|v| v.as_f64());
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    // Anthropic 的 system 是顶层字段
    let system = body.get("system")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let messages = body.get("messages")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter().filter_map(|msg| {
                let role = msg.get("role")?.as_str()?.to_string();
                // Anthropic content 可以是字符串或数组
                let content = match msg.get("content")? {
                    Value::String(s) => s.clone(),
                    Value::Array(blocks) => {
                        blocks.iter()
                            .filter_map(|b| b.get("text")?.as_str())
                            .collect::<Vec<_>>()
                            .join("")
                    }
                    _ => return None,
                };
                Some(UnifiedMessage { role, content })
            }).collect()
        })
        .unwrap_or_default();

    UnifiedRequest { model, messages, max_tokens, temperature, stream, system }
}

/// 转换为 OpenAI Chat Completions 格式
pub fn to_openai(req: &UnifiedRequest) -> Value {
    let mut messages = Vec::new();

    // 添加 system message
    if let Some(sys) = &req.system {
        messages.push(serde_json::json!({"role": "system", "content": sys}));
    }

    for msg in &req.messages {
        messages.push(serde_json::json!({
            "role": msg.role,
            "content": msg.content
        }));
    }

    let mut body = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "stream": req.stream,
    });

    if let Some(max) = req.max_tokens {
        body["max_tokens"] = serde_json::json!(max);
    }
    if let Some(temp) = req.temperature {
        body["temperature"] = serde_json::json!(temp);
    }

    body
}

/// 转换为 OpenAI Responses 格式
pub fn to_openai_responses(req: &UnifiedRequest) -> Value {
    let mut input: Vec<Value> = Vec::new();

    for msg in &req.messages {
        input.push(serde_json::json!({
            "role": msg.role,
            "content": msg.content
        }));
    }

    let mut body = serde_json::json!({
        "model": req.model,
        "input": if input.len() == 1 && input[0]["role"] == "user" {
            serde_json::Value::String(input[0]["content"].as_str().unwrap_or("").to_string())
        } else {
            serde_json::Value::Array(input)
        },
        "stream": req.stream,
    });

    if let Some(max) = req.max_tokens {
        body["max_output_tokens"] = serde_json::json!(max);
    }
    if let Some(temp) = req.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(sys) = &req.system {
        body["instructions"] = serde_json::json!(sys);
    }

    body
}

/// 转换为 Anthropic Messages 格式
pub fn to_anthropic(req: &UnifiedRequest) -> Value {
    let mut messages = Vec::new();

    for msg in &req.messages {
        messages.push(serde_json::json!({
            "role": msg.role,
            "content": msg.content
        }));
    }

    let mut body = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens.unwrap_or(4096),
        "stream": req.stream,
    });

    if let Some(temp) = req.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(sys) = &req.system {
        body["system"] = serde_json::json!(sys);
    }

    body
}

/// 统一的中间响应表示
#[derive(Debug, Clone)]
pub struct UnifiedResponse {
    pub id: String,
    pub model: String,
    pub content: String,
    pub finish_reason: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub is_error: bool,
    pub error_message: Option<String>,
}

/// 从 OpenAI Chat Completions 响应解析
pub fn parse_openai_response(body: &Value) -> UnifiedResponse {
    let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();

    // 检查错误
    if let Some(error) = body.get("error") {
        return UnifiedResponse {
            id, model,
            content: String::new(),
            finish_reason: None,
            input_tokens: 0, output_tokens: 0,
            is_error: true,
            error_message: error.get("message").and_then(|v| v.as_str()).map(|s| s.to_string()),
        };
    }

    let content = body.get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("message").or(c.get("delta")))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let finish_reason = body.get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("finish_reason"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let input_tokens = body.get("usage").and_then(|u| u.get("prompt_tokens")).and_then(|v| v.as_u64()).unwrap_or(0);
    let output_tokens = body.get("usage").and_then(|u| u.get("completion_tokens")).and_then(|v| v.as_u64()).unwrap_or(0);

    UnifiedResponse { id, model, content, finish_reason, input_tokens, output_tokens, is_error: false, error_message: None }
}

/// 从 OpenAI Responses 响应解析
pub fn parse_openai_responses_response(body: &Value) -> UnifiedResponse {
    let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();

    if let Some(error) = body.get("error") {
        return UnifiedResponse {
            id, model, content: String::new(), finish_reason: None,
            input_tokens: 0, output_tokens: 0, is_error: true,
            error_message: error.get("message").and_then(|v| v.as_str()).map(|s| s.to_string()),
        };
    }

    // Responses API 的 output 是数组
    let content = body.get("output")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    // message 类型
                    if item.get("type").and_then(|v| v.as_str()) == Some("message") {
                        item.get("content")
                            .and_then(|v| v.as_array())
                            .map(|c| {
                                c.iter()
                                    .filter_map(|block| block.get("text")?.as_str())
                                    .collect::<Vec<_>>()
                                    .join("")
                            })
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();

    let finish_reason = body.get("status")
        .and_then(|v| v.as_str())
        .map(|s| {
            match s {
                "completed" => "stop".to_string(),
                "incomplete" => "length".to_string(),
                _ => s.to_string(),
            }
        });

    let input_tokens = body.get("usage").and_then(|u| u.get("input_tokens")).and_then(|v| v.as_u64()).unwrap_or(0);
    let output_tokens = body.get("usage").and_then(|u| u.get("output_tokens")).and_then(|v| v.as_u64()).unwrap_or(0);

    UnifiedResponse { id, model, content, finish_reason, input_tokens, output_tokens, is_error: false, error_message: None }
}

/// 从 Anthropic Messages 响应解析
pub fn parse_anthropic_response(body: &Value) -> UnifiedResponse {
    let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();

    if body.get("type").and_then(|v| v.as_str()) == Some("error") {
        let error = body.get("error").unwrap_or(&Value::Null);
        return UnifiedResponse {
            id, model, content: String::new(), finish_reason: None,
            input_tokens: 0, output_tokens: 0, is_error: true,
            error_message: error.get("message").and_then(|v| v.as_str()).map(|s| s.to_string()),
        };
    }

    let content = body.get("content")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|block| block.get("text")?.as_str())
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();

    let finish_reason = body.get("stop_reason")
        .and_then(|v| v.as_str())
        .map(|s| {
            match s {
                "end_turn" => "stop".to_string(),
                "max_tokens" => "length".to_string(),
                _ => s.to_string(),
            }
        });

    let input_tokens = body.get("usage").and_then(|u| u.get("input_tokens")).and_then(|v| v.as_u64()).unwrap_or(0);
    let output_tokens = body.get("usage").and_then(|u| u.get("output_tokens")).and_then(|v| v.as_u64()).unwrap_or(0);

    UnifiedResponse { id, model, content, finish_reason, input_tokens, output_tokens, is_error: false, error_message: None }
}

/// 转换为 OpenAI Chat Completions 响应格式
pub fn to_openai_response(resp: &UnifiedResponse) -> Value {
    if resp.is_error {
        return serde_json::json!({
            "error": {
                "message": resp.error_message.as_deref().unwrap_or("未知错误"),
                "type": "server_error"
            }
        });
    }

    serde_json::json!({
        "id": format!("chatcmpl-{}", resp.id),
        "object": "chat.completion",
        "model": resp.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": resp.content
            },
            "finish_reason": resp.finish_reason.as_deref().unwrap_or("stop")
        }],
        "usage": {
            "prompt_tokens": resp.input_tokens,
            "completion_tokens": resp.output_tokens,
            "total_tokens": resp.input_tokens + resp.output_tokens
        }
    })
}

/// 转换为 OpenAI Responses 响应格式
pub fn to_openai_responses_response(resp: &UnifiedResponse) -> Value {
    if resp.is_error {
        return serde_json::json!({
            "error": {
                "message": resp.error_message.as_deref().unwrap_or("未知错误"),
                "type": "server_error"
            }
        });
    }

    serde_json::json!({
        "id": format!("resp-{}", resp.id),
        "object": "response",
        "model": resp.model,
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": resp.content
            }]
        }],
        "usage": {
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "total_tokens": resp.input_tokens + resp.output_tokens
        },
        "status": match resp.finish_reason.as_deref() {
            Some("stop") => "completed",
            Some("length") => "incomplete",
            _ => "completed",
        }
    })
}

/// 转换为 Anthropic Messages 响应格式
pub fn to_anthropic_response(resp: &UnifiedResponse) -> Value {
    if resp.is_error {
        return serde_json::json!({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": resp.error_message.as_deref().unwrap_or("未知错误")
            }
        });
    }

    serde_json::json!({
        "id": resp.id,
        "type": "message",
        "role": "assistant",
        "model": resp.model,
        "content": [{
            "type": "text",
            "text": resp.content
        }],
        "stop_reason": match resp.finish_reason.as_deref() {
            Some("stop") => "end_turn",
            Some("length") => "max_tokens",
            _ => "end_turn",
        },
        "usage": {
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens
        }
    })
}

/// 根据源格式和目标格式转换请求体
pub fn convert_request(body: &Value, from: &crate::models::ApiType, to: &crate::models::ApiType) -> Value {
    use crate::models::ApiType;

    if std::mem::discriminant(from) == std::mem::discriminant(to) {
        return body.clone();
    }

    // 先解析为统一格式
    let unified = match from {
        ApiType::OpenAI => parse_openai(body),
        ApiType::OpenAIResponses => parse_openai_responses(body),
        ApiType::Anthropic => parse_anthropic(body),
    };

    // 再转换为目标格式
    match to {
        ApiType::OpenAI => to_openai(&unified),
        ApiType::OpenAIResponses => to_openai_responses(&unified),
        ApiType::Anthropic => to_anthropic(&unified),
    }
}

/// 根据源格式和目标格式转换响应体（非流式）
pub fn convert_response(body: &Value, from: &crate::models::ApiType, to: &crate::models::ApiType) -> Value {
    use crate::models::ApiType;

    if std::mem::discriminant(from) == std::mem::discriminant(to) {
        return body.clone();
    }

    // 先解析为统一格式
    let unified = match from {
        ApiType::OpenAI => parse_openai_response(body),
        ApiType::OpenAIResponses => parse_openai_responses_response(body),
        ApiType::Anthropic => parse_anthropic_response(body),
    };

    // 再转换为目标格式
    match to {
        ApiType::OpenAI => to_openai_response(&unified),
        ApiType::OpenAIResponses => to_openai_responses_response(&unified),
        ApiType::Anthropic => to_anthropic_response(&unified),
    }
}

/// 根据目标格式转换路径
pub fn convert_path(path: &str, from: &crate::models::ApiType, to: &crate::models::ApiType) -> String {
    use crate::models::ApiType;

    if std::mem::discriminant(from) == std::mem::discriminant(to) {
        return path.to_string();
    }

    // 从源格式的路径转换为目标格式的路径
    match to {
        ApiType::OpenAI => "chat/completions".to_string(),
        ApiType::OpenAIResponses => "responses".to_string(),
        ApiType::Anthropic => "messages".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_to_anthropic() {
        let openai_body = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        });

        let unified = parse_openai(&openai_body);
        let anthropic_body = to_anthropic(&unified);

        assert_eq!(anthropic_body["model"], "gpt-4");
        assert_eq!(anthropic_body["system"], "You are helpful.");
        assert_eq!(anthropic_body["messages"][0]["role"], "user");
        assert_eq!(anthropic_body["messages"][0]["content"], "Hello");
        assert_eq!(anthropic_body["max_tokens"], 100);
    }

    #[test]
    fn test_openai_responses_to_openai() {
        let responses_body = serde_json::json!({
            "model": "mimo-v2-pro",
            "input": "Hello",
            "max_output_tokens": 100
        });

        let unified = parse_openai_responses(&responses_body);
        let openai_body = to_openai(&unified);

        assert_eq!(openai_body["model"], "mimo-v2-pro");
        assert_eq!(openai_body["messages"][0]["role"], "user");
        assert_eq!(openai_body["messages"][0]["content"], "Hello");
        assert_eq!(openai_body["max_tokens"], 100);
    }

    #[test]
    fn test_anthropic_to_openai() {
        let anthropic_body = serde_json::json!({
            "model": "claude-3",
            "system": "Be helpful",
            "messages": [
                {"role": "user", "content": "Hi"}
            ],
            "max_tokens": 200
        });

        let unified = parse_anthropic(&anthropic_body);
        let openai_body = to_openai(&unified);

        assert_eq!(openai_body["model"], "claude-3");
        assert_eq!(openai_body["messages"][0]["role"], "system");
        assert_eq!(openai_body["messages"][0]["content"], "Be helpful");
        assert_eq!(openai_body["messages"][1]["role"], "user");
        assert_eq!(openai_body["messages"][1]["content"], "Hi");
    }
}
