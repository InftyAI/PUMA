build:
	cargo build && cp target/debug/puma ./puma

test:
	cargo test
