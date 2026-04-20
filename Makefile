build:
	cargo build && cp target/debug/puma ./puma

test:
	cargo test

lint:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings

format:
	cargo fmt --all
	cargo clippy --fix --allow-dirty