// lib.rs - internal modules for the PUMA binary
// Modules are private as PUMA is primarily a CLI tool, not a library

#![allow(dead_code)]

mod api;
mod backend;
mod registry;
mod storage;
mod utils;
