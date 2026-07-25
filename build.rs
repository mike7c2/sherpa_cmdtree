fn main() {
    // The `sherpa` feature links sherpa-onnx dynamically. sherpa-rs-sys copies
    // libsherpa-onnx-c-api.so next to our binary but adds no rpath on Linux, so a
    // standalone (non-`cargo run`) binary can't find it. Stamp an `$ORIGIN` rpath
    // so the binary looks for the lib in its own directory — ship the .so alongside
    // the executable (as it already sits in target/<profile>/).
    let sherpa = std::env::var_os("CARGO_FEATURE_SHERPA").is_some();
    let linux = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
    if sherpa && linux {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN");
    }
}
