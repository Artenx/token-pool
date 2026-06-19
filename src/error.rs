use actix_web::{HttpResponse, ResponseError};
use std::fmt;

#[derive(Debug)]
pub enum AppError {
    NotFound(String),
    Unauthorized,
    BadRequest(String),
    Internal(String),
    Proxy(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::NotFound(msg) => write!(f, "未找到: {}", msg),
            AppError::Unauthorized => write!(f, "未授权"),
            AppError::BadRequest(msg) => write!(f, "请求错误: {}", msg),
            AppError::Internal(msg) => write!(f, "内部错误: {}", msg),
            AppError::Proxy(msg) => write!(f, "代理错误: {}", msg),
        }
    }
}

impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        let body = serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": match self {
                    AppError::Unauthorized => "authentication_error",
                    AppError::BadRequest(_) => "invalid_request_error",
                    AppError::NotFound(_) => "not_found_error",
                    _ => "server_error",
                }
            }
        });
        match self {
            AppError::Unauthorized => HttpResponse::Unauthorized().json(body),
            AppError::NotFound(_) => HttpResponse::NotFound().json(body),
            AppError::BadRequest(_) => HttpResponse::BadRequest().json(body),
            _ => HttpResponse::InternalServerError().json(body),
        }
    }
}
