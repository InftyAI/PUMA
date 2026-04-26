pub mod chat;
pub mod completions;
pub mod models;
pub mod routes;
pub mod types;

pub use routes::create_router;

#[cfg(test)]
mod tests;
