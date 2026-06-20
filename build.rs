use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/onnx.proto");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let repo_proto = manifest_dir.join("proto").join("onnx.proto");

    prost_build::Config::new()
        .bytes(["."])
        .compile_protos(&[&repo_proto], &[&manifest_dir])?;

    Ok(())
}
