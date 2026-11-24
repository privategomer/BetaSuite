fn main() {
    // One version for engine, UI and installers: the repository's VERSION file.
    let version =
        std::fs::read_to_string("../../VERSION").expect("VERSION file at the repository root");
    println!("cargo:rustc-env=BETASUITE_VERSION={}", version.trim());
    println!("cargo:rerun-if-changed=../../VERSION");
    tauri_build::build();
}
