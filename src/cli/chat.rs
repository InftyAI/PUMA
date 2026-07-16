use rustyline::error::ReadlineError;
use rustyline::hint::{Hint, Hinter};
use rustyline::{Context, Editor};
use rustyline_derive::{Completer, Helper, Highlighter, Validator};
use std::io::{self, Write};
use tokio_stream::StreamExt;

use crate::backend::{Backend, LLMEngine};

#[derive(Clone)]
struct PlaceholderHint {
    display: String,
}

impl Hint for PlaceholderHint {
    fn display(&self) -> &str {
        &self.display
    }

    fn completion(&self) -> Option<&str> {
        None
    }
}

/// Hint helper that shows placeholder when input is empty
#[derive(Helper, Completer, Highlighter, Validator)]
struct PlaceholderHinter {
    placeholder: String,
}

impl Hinter for PlaceholderHinter {
    type Hint = PlaceholderHint;

    fn hint(&self, line: &str, _pos: usize, _ctx: &Context<'_>) -> Option<PlaceholderHint> {
        if line.is_empty() {
            // Grey/dimmed color: \x1b[2m ... \x1b[0m
            Some(PlaceholderHint {
                display: format!("\x1b[2m{}\x1b[0m", self.placeholder),
            })
        } else {
            None
        }
    }
}

/// Interactive chat loop for puma run
pub async fn interactive_chat<E: InferenceEngine>(
    engine: &E,
    model: &str,
) -> Result<(), io::Error> {
    let mut conversation_history = Vec::new();

    // Setup editor with placeholder hinter
    let helper = PlaceholderHinter {
        placeholder: "Send a message (Ctrl-C or 'exit' to quit)".to_string(),
    };
    let mut rl = Editor::<PlaceholderHinter, rustyline::history::DefaultHistory>::new()
        .map_err(io::Error::other)?;
    rl.set_helper(Some(helper));

    loop {
        let readline = rl.readline("> ");

        let input = match readline {
            Ok(line) => line.trim().to_string(),
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                return Err(io::Error::other(err));
            }
        };

        // Exit commands
        if input.is_empty() {
            continue;
        }
        if input == "exit" {
            break;
        }

        // Add user message to history
        conversation_history.push(format!("User: {}", input));

        // Build prompt from conversation history
        let prompt = conversation_history.join("\n") + "\nAssistant:";

        // Empty line before response
        println!();

        // Generate response with streaming
        match engine.generate_stream(model, &prompt, 512, 0.7).await {
            Ok(mut stream) => {
                let mut full_response = String::new();

                // Display tokens as they arrive
                while let Some(token) = stream.next().await {
                    print!("{}", token);
                    io::stdout().flush()?;
                    full_response.push_str(&token);
                }

                println!("\n"); // Double newline after response

                // Add assistant response to history
                conversation_history.push(format!("Assistant: {}", full_response.trim()));
            }
            Err(e) => {
                eprintln!("Error: {}\n", e);
            }
        }
    }

    Ok(())
}
