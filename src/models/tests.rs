use super::*;
use std::cell::Cell;
use std::ffi::OsString;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Barrier, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::Instant;

static MODEL_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Debug)]
struct TestRequest {
    request_line: String,
    headers: Vec<(String, String)>,
}

impl TestRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

struct TestResponse {
    status: u16,
    reason: &'static str,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    content_length: TestContentLength,
}

enum TestContentLength {
    Automatic,
    Omitted,
    Raw(String),
}

impl TestResponse {
    fn ok(body: impl Into<Vec<u8>>) -> Self {
        Self {
            status: 200,
            reason: "OK",
            headers: Vec::new(),
            body: body.into(),
            content_length: TestContentLength::Automatic,
        }
    }

    fn with_header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers.push((name.to_string(), value.into()));
        self
    }

    fn without_content_length(mut self) -> Self {
        self.content_length = TestContentLength::Omitted;
        self
    }

    fn with_raw_content_length(mut self, value: impl Into<String>) -> Self {
        self.content_length = TestContentLength::Raw(value.into());
        self
    }
}

struct ModelDirGuard {
    previous: Option<OsString>,
}

struct EnvVarGuard {
    name: &'static str,
    previous: Option<OsString>,
}

impl ModelDirGuard {
    fn set(path: &Path) -> Self {
        let previous = std::env::var_os("DENOIZE_MODEL_DIR");
        std::env::set_var("DENOIZE_MODEL_DIR", path);
        Self { previous }
    }
}

impl Drop for ModelDirGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            std::env::set_var("DENOIZE_MODEL_DIR", previous);
        } else {
            std::env::remove_var("DENOIZE_MODEL_DIR");
        }
    }
}

impl EnvVarGuard {
    fn set(name: &'static str, value: Option<&str>) -> Self {
        let previous = std::env::var_os(name);
        if let Some(value) = value {
            std::env::set_var(name, value);
        } else {
            std::env::remove_var(name);
        }
        Self { name, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            std::env::set_var(self.name, previous);
        } else {
            std::env::remove_var(self.name);
        }
    }
}

fn lock_model_environment() -> MutexGuard<'static, ()> {
    MODEL_ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(unix)]
fn create_fifo(path: &Path) {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
}

fn test_sha256(bytes: &[u8]) -> &'static str {
    let mut digest = Sha256::new();
    digest.update(bytes);
    Box::leak(format!("{:x}", digest.finalize()).into_boxed_str())
}

fn test_model(bytes: &[u8]) -> ModelInfo {
    ModelInfo {
        name: "model-download-test",
        backend: "model-download-test",
        filename: "model.onnx",
        url: "http://127.0.0.1/unused",
        revision: "0000000000000000000000000000000000000000",
        sha256: test_sha256(bytes),
        size_bytes: bytes.len() as u64,
        license: "MIT",
        sample_rate: 16_000,
    }
}

fn test_catalog_model(bytes: &[u8]) -> CatalogModel {
    let catalog = embedded_catalog();
    CatalogModel {
        name: "catalog-model-test".into(),
        backend: "catalog-model-test".into(),
        filename: "catalog-model.onnx".into(),
        url: "https://models.example.test/catalog-model.onnx".into(),
        revision: "catalog-test-revision".into(),
        sha256: test_sha256(bytes).into(),
        size_bytes: bytes.len() as u64,
        license: "MIT".into(),
        sample_rate: 16_000,
        catalog: catalog.identity().clone(),
    }
}

fn direct_options() -> ModelDownloadOptions {
    ModelDownloadOptions {
        proxy: ModelProxy::Disabled,
        connect_timeout: Duration::from_secs(2),
        read_timeout: Duration::from_secs(2),
        request_timeout: Duration::from_secs(5),
        ..Default::default()
    }
}

fn read_test_request(stream: &mut TcpStream) -> TestRequest {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 1024];
    while !bytes.windows(4).any(|window| window == b"\r\n\r\n") {
        let count = stream.read(&mut buffer).unwrap();
        assert!(count > 0, "client closed before completing HTTP headers");
        bytes.extend_from_slice(&buffer[..count]);
        assert!(
            bytes.len() < 64 * 1024,
            "test request headers are too large"
        );
    }
    let text = String::from_utf8(bytes).unwrap();
    let mut lines = text.split("\r\n");
    let request_line = lines.next().unwrap().to_string();
    let headers = lines
        .take_while(|line| !line.is_empty())
        .map(|line| {
            let (name, value) = line.split_once(':').unwrap();
            (name.trim().to_string(), value.trim().to_string())
        })
        .collect();
    TestRequest {
        request_line,
        headers,
    }
}

fn write_test_response(stream: &mut TcpStream, response: TestResponse) {
    let mut head = format!("HTTP/1.1 {} {}\r\n", response.status, response.reason);
    match response.content_length {
        TestContentLength::Automatic => {
            head.push_str(&format!("Content-Length: {}\r\n", response.body.len()));
        }
        TestContentLength::Omitted => {}
        TestContentLength::Raw(value) => {
            head.push_str(&format!("Content-Length: {value}\r\n"));
        }
    }
    head.push_str("Connection: close\r\n");
    for (name, value) in response.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&response.body);
    let _ = stream.flush();
}

fn spawn_test_server<F>(
    expected_requests: usize,
    handler: F,
) -> (String, JoinHandle<Vec<TestRequest>>)
where
    F: Fn(usize, &TestRequest) -> TestResponse + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut requests = Vec::new();
        while requests.len() < expected_requests && Instant::now() < deadline {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let request = read_test_request(&mut stream);
                    let response = handler(requests.len(), &request);
                    write_test_response(&mut stream, response);
                    requests.push(request);
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("test server accept failed: {error}"),
            }
        }
        requests
    });
    (format!("http://{address}/model.onnx"), handle)
}

fn download_paths(directory: &Path) -> (PathBuf, PathBuf) {
    (
        directory.join("download.part"),
        directory.join("download.part.meta"),
    )
}

fn download_for_test<C, P>(
    model: &ModelInfo,
    source_url: &str,
    options: &ModelDownloadOptions,
    directory: &Path,
    cancelled: &mut C,
    progress: &mut P,
) -> Result<(), String>
where
    C: FnMut() -> bool,
    P: FnMut(u64, Option<u64>),
{
    let source = Url::parse(source_url).unwrap();
    let source_id = source_identity(source_url);
    let (partial, metadata) = download_paths(directory);
    download(
        model, options, &source, &source_id, &partial, &metadata, cancelled, progress,
    )
}

fn seed_partial(
    directory: &Path,
    source_url: &str,
    model: &ModelInfo,
    bytes: &[u8],
    etag: Option<&str>,
    total: Option<u64>,
) {
    let (partial, metadata_path) = download_paths(directory);
    std::fs::write(partial, bytes).unwrap();
    write_metadata(
        &metadata_path,
        &PartialMetadata {
            version: PARTIAL_METADATA_VERSION,
            source_id: source_identity(source_url),
            expected_sha256: model.sha256.to_string(),
            etag: etag.map(str::to_string),
            last_modified: None,
            total,
        },
    )
    .unwrap();
}

fn request_range_start(request: &TestRequest) -> usize {
    request
        .header("Range")
        .and_then(|value| value.strip_prefix("bytes="))
        .and_then(|value| value.strip_suffix('-'))
        .and_then(|value| value.parse().ok())
        .expect("expected a byte range request")
}

fn assert_manifest_progress(progress: &[(u64, Option<u64>)], expected_size: u64) {
    assert!(!progress.is_empty());
    assert!(progress
        .iter()
        .all(|(downloaded, total)| *downloaded <= expected_size && *total == Some(expected_size)));
}

#[test]
fn manifest_has_pinned_integrity_and_metadata() {
    for model in MODELS {
        assert_eq!(model.sha256.len(), 64);
        assert!(model.size_bytes > 0);
        assert_eq!(model.revision.len(), 40);
        assert!(model.url.contains(model.revision));
        assert!(model.sample_rate > 0);
        assert!(!model.license.is_empty());
        assert!(ModelSpec::legacy(model).catalog.is_some());
    }
    assert_eq!(find("gtcrn-dns3").unwrap().size_bytes, 535_190);
}

#[test]
fn verification_requires_the_manifest_size_and_checksum() {
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("model.onnx");
    let payload = b"verified cache bytes";
    let model = test_model(payload);

    std::fs::write(&destination, payload).unwrap();
    assert_eq!(verify_at(&model, &destination).unwrap(), destination);

    std::fs::write(&destination, &payload[..payload.len() - 1]).unwrap();
    let error = verify_at(&model, &destination).unwrap_err();
    assert!(error.contains("size mismatch"), "unexpected error: {error}");

    let mut oversized = payload.to_vec();
    oversized.push(0);
    std::fs::write(&destination, oversized).unwrap();
    let error = verify_at(&model, &destination).unwrap_err();
    assert!(error.contains("size mismatch"), "unexpected error: {error}");

    let mut corrupt = payload.to_vec();
    corrupt[0] ^= 1;
    std::fs::write(&destination, corrupt).unwrap();
    let error = verify_at(&model, &destination).unwrap_err();
    assert!(
        error.contains("checksum mismatch"),
        "unexpected error: {error}"
    );
}

#[test]
fn removal_is_idempotent() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.onnx");
    std::fs::write(&path, b"model").unwrap();
    assert!(remove_file_if_present(&path).unwrap());
    assert!(!remove_file_if_present(&path).unwrap());
}

#[test]
fn content_range_parser_is_strict() {
    assert_eq!(
        parse_satisfied_content_range("bytes 10-19/20"),
        Some((10, 19, Some(20)))
    );
    assert_eq!(
        parse_satisfied_content_range("bytes 10-19/*"),
        Some((10, 19, None))
    );
    assert_eq!(parse_unsatisfied_content_range("bytes */20"), Some(20));
    assert!(parse_satisfied_content_range("bytes 20-19/20").is_none());
    assert!(parse_satisfied_content_range("bytes 10-20/20").is_none());
    assert!(parse_satisfied_content_range("bytes 0-18446744073709551615/*").is_none());
    assert!(parse_satisfied_content_range("items 10-19/20").is_none());
}

#[test]
fn oversized_partial_metadata_is_ignored() {
    let directory = tempfile::tempdir().unwrap();
    let metadata = directory.path().join("model.onnx.part.meta");
    std::fs::write(
        &metadata,
        vec![b'x'; (MAX_PARTIAL_METADATA_BYTES + 1) as usize],
    )
    .unwrap();
    assert!(load_matching_metadata(&metadata, "source", "checksum", 1)
        .unwrap()
        .is_none());
}

#[test]
fn no_proxy_matches_hosts_domains_and_ports() {
    assert!(no_proxy_matches("example.com", Some(443), "example.com"));
    assert!(no_proxy_matches(
        "api.example.com",
        Some(443),
        ".example.com"
    ));
    assert!(no_proxy_matches(
        "api.example.com",
        Some(8443),
        "example.com:8443"
    ));
    assert!(!no_proxy_matches(
        "api.example.com",
        Some(443),
        "example.com:8443"
    ));
    assert!(!no_proxy_matches(
        "badexample.com",
        Some(443),
        "example.com"
    ));
    assert!(no_proxy_matches("anything.invalid", None, "*"));
    assert!(no_proxy_matches("::1", Some(80), "[::1]:80"));
    assert!(no_proxy_matches("127.1.2.3", Some(80), "127.0.0.0/8"));
    assert!(!no_proxy_matches("128.1.2.3", Some(80), "127.0.0.0/8"));
}

#[test]
fn diagnostics_redact_url_credentials_and_query() {
    let value = redact_url("https://user:secret@example.com/model.onnx?token=secret#fragment");
    assert_eq!(value, "https://example.com/model.onnx");
    assert!(!value.contains("secret"));
    let options = ModelDownloadOptions {
        source_url: Some("https://user:secret@example.com/model.onnx?token=secret".to_string()),
        proxy: ModelProxy::Url("proxy-user:proxy-secret@proxy:8080".into()),
        authentication: Some(ModelAuthentication::Bearer("bearer-secret".into())),
        ..Default::default()
    };
    let debug = format!("{options:?}");
    assert!(!debug.contains("secret"));
    assert!(!debug.contains("bearer-secret"));
    let proxy_debug = format!("{:?}", options.proxy);
    assert!(!proxy_debug.contains("proxy-user"));
    assert!(!proxy_debug.contains("proxy-secret"));
    let basic = format!(
        "{:?}",
        ModelAuthentication::Basic {
            username: "private-user@example.test".into(),
            password: "private-password".into(),
        }
    );
    assert!(!basic.contains("private-user"));
    assert!(!basic.contains("private-password"));
}

#[test]
fn signed_catalog_import_enforces_rollback_equivocation_and_key_rotation() {
    let _environment = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());

    let fixtures = [
        (
            "seq2",
            include_bytes!("testdata/catalog-seq2.json").as_slice(),
            include_bytes!("testdata/catalog-seq2.json.sig").as_slice(),
        ),
        (
            "seq2-alt",
            include_bytes!("testdata/catalog-seq2-alt.json").as_slice(),
            include_bytes!("testdata/catalog-seq2-alt.json.sig").as_slice(),
        ),
        (
            "seq3",
            include_bytes!("testdata/catalog-seq3.json").as_slice(),
            include_bytes!("testdata/catalog-seq3.json.sig").as_slice(),
        ),
        (
            "seq4",
            include_bytes!("testdata/catalog-seq4.json").as_slice(),
            include_bytes!("testdata/catalog-seq4.json.sig").as_slice(),
        ),
    ];
    let mut paths = Vec::new();
    for (name, catalog, signature) in fixtures {
        let catalog_path = directory.path().join(format!("{name}.json"));
        let signature_path = directory.path().join(format!("{name}.json.sig"));
        std::fs::write(&catalog_path, catalog).unwrap();
        std::fs::write(&signature_path, signature).unwrap();
        paths.push((catalog_path, signature_path));
    }

    assert_eq!(active_catalog().unwrap().sequence(), 1);
    assert_eq!(
        import_catalog(&paths[0].0, &paths[0].1).unwrap().sequence(),
        2
    );
    let error = import_catalog(&paths[1].0, &paths[1].1).unwrap_err();
    assert!(error.contains("different model catalog content"), "{error}");

    assert_eq!(
        import_catalog(&paths[2].0, &paths[2].1).unwrap().sequence(),
        3
    );
    let error = import_catalog(&paths[0].0, &paths[0].1).unwrap_err();
    assert!(error.contains("rollback from sequence 3 to 2"), "{error}");

    let active_path = directory.path().join(".catalog/active.json");
    std::fs::remove_file(&active_path).unwrap();
    let error = active_catalog().unwrap_err();
    assert!(error.contains("signed cache is missing"), "{error}");
    assert_eq!(
        import_catalog(&paths[2].0, &paths[2].1).unwrap().sequence(),
        3
    );

    let rotated = import_catalog(&paths[3].0, &paths[3].1).unwrap();
    assert_eq!(rotated.sequence(), 4);
    assert_eq!(rotated.signing_key_id(), "557E67D5F983C071");
    let status = catalog_status().unwrap();
    assert_eq!(status.sequence, 4);
    assert_eq!(status.highest_accepted_sequence, 4);
    assert!(matches!(status.origin, CatalogOrigin::Signed { .. }));
}

#[test]
fn newer_embedded_catalog_persists_its_rollback_floor() {
    let _environment = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let sequence_two = directory.path().join("sequence-two.json");
    let sequence_two_signature = directory.path().join("sequence-two.json.sig");
    std::fs::write(&sequence_two, include_bytes!("testdata/catalog-seq2.json")).unwrap();
    std::fs::write(
        &sequence_two_signature,
        include_bytes!("testdata/catalog-seq2.json.sig"),
    )
    .unwrap();
    import_catalog(&sequence_two, &sequence_two_signature).unwrap();

    // A newer trusted binary must supersede even an obsolete corrupt envelope,
    // then durably raise the floor before returning its embedded catalog.
    std::fs::write(directory.path().join(".catalog/active.json"), b"corrupt").unwrap();
    let embedded_sequence_three = catalog::parse_catalog(
        include_bytes!("testdata/catalog-seq3.json"),
        CatalogOrigin::Embedded,
    )
    .unwrap();
    let expected_sha256 = embedded_sequence_three.sha256().to_string();
    let promoted = catalog::promote_embedded_catalog(embedded_sequence_three).unwrap();
    assert_eq!(promoted.sequence(), 3);

    let state: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.path().join(".catalog/state.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(state["highest_sequence"], 3);
    assert_eq!(state["catalog_sha256"], expected_sha256);
}

#[cfg(unix)]
#[test]
fn catalog_storage_rejects_a_symlinked_state_directory() {
    use std::os::unix::fs::symlink;

    let _environment = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let redirected = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    symlink(redirected.path(), directory.path().join(".catalog")).unwrap();

    let error = active_catalog().unwrap_err();
    assert!(error.contains("symbolic link for model state"), "{error}");
}

#[cfg(unix)]
#[test]
fn catalog_import_rejects_fifo_inputs_without_blocking() {
    let _environment = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let catalog_fifo = directory.path().join("catalog.fifo");
    let signature_fifo = directory.path().join("catalog.sig.fifo");
    let regular_catalog = directory.path().join("catalog.json");
    let regular_signature = directory.path().join("catalog.json.sig");
    create_fifo(&catalog_fifo);
    create_fifo(&signature_fifo);
    std::fs::write(
        &regular_catalog,
        include_bytes!("testdata/catalog-seq2.json"),
    )
    .unwrap();
    std::fs::write(
        &regular_signature,
        include_bytes!("testdata/catalog-seq2.json.sig"),
    )
    .unwrap();

    let started = Instant::now();
    let error = import_catalog(&catalog_fifo, &regular_signature).unwrap_err();
    assert!(error.contains("not a regular file"), "{error}");
    assert!(started.elapsed() < std::time::Duration::from_secs(1));

    let started = Instant::now();
    let error = import_catalog(&regular_catalog, &signature_fifo).unwrap_err();
    assert!(error.contains("not a regular file"), "{error}");
    assert!(started.elapsed() < std::time::Duration::from_secs(1));
}

#[test]
fn catalog_model_install_records_validates_and_removes_provenance() {
    let _environment = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let bytes = b"catalog model provenance bytes";
    let model = test_catalog_model(bytes);
    let source = directory.path().join("source.onnx");
    std::fs::write(&source, bytes).unwrap();

    let installed = install_catalog_model_from_file(&model, &source).unwrap();
    assert_eq!(verify_catalog_model(&model).unwrap(), installed);
    let provenance = catalog_model_provenance(&model).unwrap();
    assert_eq!(provenance.model_name, model.name);
    assert_eq!(provenance.artifact_sha256, model.sha256);
    assert_eq!(provenance.catalog_sha256, model.catalog_sha256());
    assert_eq!(
        provenance.installation_source,
        ModelInstallationSource::LocalFile
    );

    let spec = ModelSpec::catalog(&model);
    let provenance_path = provenance_path(&spec, &installed).unwrap();
    let mut tampered: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&provenance_path).unwrap()).unwrap();
    tampered["revision"] = serde_json::Value::String("tampered".into());
    std::fs::write(
        &provenance_path,
        serde_json::to_vec_pretty(&tampered).unwrap(),
    )
    .unwrap();
    let error = verify_catalog_model(&model).unwrap_err();
    assert!(error.contains("provenance does not match"), "{error}");

    tampered["revision"] = serde_json::Value::String(model.revision.clone());
    tampered["installation_source"] = serde_json::json!({
        "kind": "alternate-url",
        "url": "https://user:secret@models.example.test/model.onnx?token=secret"
    });
    std::fs::write(
        &provenance_path,
        serde_json::to_vec_pretty(&tampered).unwrap(),
    )
    .unwrap();
    let error = verify_catalog_model(&model).unwrap_err();
    assert!(error.contains("provenance does not match"), "{error}");

    tampered["installation_source"] = serde_json::json!({ "kind": "local-file" });
    tampered["catalog_origin"] = serde_json::json!({
        "kind": "signed",
        "source": "https://user:secret@models.example.test/catalog.json?token=secret"
    });
    std::fs::write(
        &provenance_path,
        serde_json::to_vec_pretty(&tampered).unwrap(),
    )
    .unwrap();
    let error = verify_catalog_model(&model).unwrap_err();
    assert!(error.contains("provenance does not match"), "{error}");

    tampered["catalog_origin"] = serde_json::json!({ "kind": "embedded", "source": "forged" });
    std::fs::write(
        &provenance_path,
        serde_json::to_vec_pretty(&tampered).unwrap(),
    )
    .unwrap();
    let error = verify_catalog_model(&model).unwrap_err();
    assert!(error.contains("invalid model provenance"), "{error}");

    tampered["catalog_origin"] = serde_json::json!({ "kind": "embedded" });
    std::fs::write(
        &provenance_path,
        serde_json::to_vec_pretty(&tampered).unwrap(),
    )
    .unwrap();
    let mut same_catalog_from_another_origin = model.clone();
    same_catalog_from_another_origin.catalog.origin = CatalogOrigin::Signed {
        source: "local-import".into(),
    };
    assert_eq!(
        verify_catalog_model(&same_catalog_from_another_origin).unwrap(),
        installed
    );

    assert!(remove_catalog_model(&model).unwrap());
    assert!(!installed.exists());
    assert!(!provenance_path.exists());
}

#[test]
fn cancelled_catalog_install_leaves_no_model_or_partial_provenance() {
    let _environment = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let bytes = b"cancelled catalog model";
    let model = test_catalog_model(bytes);
    let source = directory.path().join("source.onnx");
    std::fs::write(&source, bytes).unwrap();

    let error = install_catalog_model_from_file_with_progress(&model, &source, || true, |_, _| {})
        .unwrap_err();
    assert_eq!(error, "cancelled");
    let destination = path_for_catalog_model(&model).unwrap();
    assert!(!destination.exists());
    assert!(!destination.parent().unwrap().join(".provenance").exists());
}

#[test]
fn loopback_origin_credentials_are_rejected_when_an_http_proxy_is_selected() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let model = test_model(b"proxy must not receive origin credentials");
    let directory = tempfile::tempdir().unwrap();
    let options = ModelDownloadOptions {
        proxy: ModelProxy::Url(format!("http://{}", listener.local_addr().unwrap())),
        authentication: Some(ModelAuthentication::Bearer("origin-secret".into())),
        connect_timeout: Duration::from_millis(100),
        ..Default::default()
    };

    let error = download_for_test(
        &model,
        "http://127.0.0.1:9/model.onnx",
        &options,
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap_err();

    assert!(
        error.contains("non-HTTPS transport"),
        "unexpected error: {error}"
    );
    assert!(matches!(listener.accept(), Err(error) if error.kind() == ErrorKind::WouldBlock));
}

#[test]
fn signed_http_source_is_rejected_before_contacting_a_proxy() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let model = test_model(b"proxy must not receive a signed source URL");
    let directory = tempfile::tempdir().unwrap();
    let options = ModelDownloadOptions {
        proxy: ModelProxy::Url(format!("http://{}", listener.local_addr().unwrap())),
        ..Default::default()
    };

    let error = download_for_test(
        &model,
        "http://127.0.0.1:9/model.onnx?signature=origin-secret",
        &options,
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap_err();

    assert!(
        error.contains("non-HTTPS transport"),
        "unexpected error: {error}"
    );
    assert!(!error.contains("origin-secret"));
    assert!(matches!(listener.accept(), Err(error) if error.kind() == ErrorKind::WouldBlock));
}

#[test]
fn offline_cache_miss_opens_no_connection() {
    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let source_url = format!("http://{}/model.onnx", listener.local_addr().unwrap());
    let model = test_model(b"offline-model");
    let options = ModelDownloadOptions {
        offline: true,
        source_url: Some(source_url),
        ..direct_options()
    };

    let error = install_with_options(&model, &options).unwrap_err();

    assert!(error.contains("offline mode"), "unexpected error: {error}");
    assert!(matches!(listener.accept(), Err(error) if error.kind() == ErrorKind::WouldBlock));
}

#[test]
fn offline_mode_uses_verified_cache_without_connecting() {
    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let payload = b"already cached offline model";
    let model = test_model(payload);
    let destination = path(&model).unwrap();
    std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
    std::fs::write(&destination, payload).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let options = ModelDownloadOptions {
        offline: true,
        source_url: Some(format!(
            "http://{}/model.onnx",
            listener.local_addr().unwrap()
        )),
        ..direct_options()
    };

    assert_eq!(install_with_options(&model, &options).unwrap(), destination);
    assert!(matches!(listener.accept(), Err(error) if error.kind() == ErrorKind::WouldBlock));
}

#[test]
fn offline_mode_promotes_a_completed_verified_partial() {
    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let payload = b"completed offline partial model";
    let model = test_catalog_model(payload);
    let destination = path_for_catalog_model(&model).unwrap();
    std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
    std::fs::write(sidecar(&destination, ".part"), payload).unwrap();
    let options = ModelDownloadOptions {
        offline: true,
        ..direct_options()
    };

    assert_eq!(
        install_catalog_model_with_options(&model, &options).unwrap(),
        destination
    );
    assert_eq!(std::fs::read(&destination).unwrap(), payload);
    assert!(!sidecar(&destination, ".part").exists());
    assert_eq!(
        catalog_model_provenance(&model)
            .unwrap()
            .installation_source,
        ModelInstallationSource::CompletedPartial
    );
}

#[test]
fn offline_mode_discards_oversized_and_complete_corrupt_partials() {
    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let payload = b"manifest bounded offline partial";
    let model = test_model(payload);
    let destination = path(&model).unwrap();
    std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
    let partial = sidecar(&destination, ".part");
    let metadata = sidecar(&destination, ".part.meta");
    let mut corrupt = payload.to_vec();
    corrupt[0] ^= 1;

    for partial_bytes in [[payload.as_slice(), b"x"].concat(), corrupt] {
        std::fs::write(&partial, partial_bytes).unwrap();
        std::fs::write(&metadata, b"stale metadata").unwrap();

        let error = install_with_options(
            &model,
            &ModelDownloadOptions {
                offline: true,
                ..direct_options()
            },
        )
        .unwrap_err();

        assert!(error.contains("offline mode"), "unexpected error: {error}");
        assert!(!destination.exists());
        assert!(!partial.exists());
        assert!(!metadata.exists());
    }
}

#[test]
fn authentication_rejects_insecure_or_malformed_credentials() {
    let remote = Url::parse("http://models.example.test/model.onnx").unwrap();
    let bearer = ModelAuthentication::Bearer("secret".into());
    validate_authentication(&remote, Some(&bearer)).unwrap();
    assert!(validate_auth_transport(&remote, true, false)
        .unwrap_err()
        .contains("non-HTTPS"));

    let loopback = Url::parse("http://127.0.0.1/model.onnx").unwrap();
    let malformed = ModelAuthentication::Bearer("secret\r\ninjected: true".into());
    assert!(validate_authentication(&loopback, Some(&malformed))
        .unwrap_err()
        .contains("invalid characters"));

    let ipv6_loopback = Url::parse("http://[::1]/model.onnx?signature=local").unwrap();
    validate_auth_transport(&ipv6_loopback, true, false).unwrap();
}

#[test]
fn local_import_accepts_matching_bytes_and_rejects_mismatch() {
    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let expected = b"verified local model";
    let model = test_model(expected);
    let valid_source = directory.path().join("valid.onnx");
    let invalid_source = directory.path().join("invalid.onnx");
    let short_source = directory.path().join("short.onnx");
    let oversized_source = directory.path().join("oversized.onnx");
    std::fs::write(&valid_source, expected).unwrap();
    std::fs::write(&invalid_source, b"tampered local model").unwrap();
    std::fs::write(&short_source, &expected[..expected.len() - 1]).unwrap();
    std::fs::write(&oversized_source, [expected.as_slice(), b"x"].concat()).unwrap();

    let installed = install_from_file(&model, &valid_source).unwrap();
    assert_eq!(std::fs::read(&installed).unwrap(), expected);
    let error = provenance(&model).unwrap_err();
    assert!(error.contains("provenance is unavailable"), "{error}");
    assert!(!installed.parent().unwrap().join(".provenance").exists());
    let partial = sidecar(&installed, ".part");
    let metadata = sidecar(&installed, ".part.meta");
    std::fs::write(&partial, b"preserved partial").unwrap();
    std::fs::write(&metadata, b"preserved metadata").unwrap();

    let error = install_from_file(&model, &invalid_source).unwrap_err();
    assert!(
        error.contains("checksum mismatch"),
        "unexpected error: {error}"
    );
    for source in [&short_source, &oversized_source] {
        let error = install_from_file(&model, source).unwrap_err();
        assert!(error.contains("size mismatch"), "unexpected error: {error}");
    }
    assert_eq!(std::fs::read(&installed).unwrap(), expected);
    assert_eq!(std::fs::read(partial).unwrap(), b"preserved partial");
    assert_eq!(std::fs::read(metadata).unwrap(), b"preserved metadata");
}

#[test]
fn local_import_reports_stale_sidecar_errors_before_publishing() {
    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let payload = b"verified local model";
    let model = test_model(payload);
    let source = directory.path().join("source.onnx");
    std::fs::write(&source, payload).unwrap();
    let destination = path(&model).unwrap();
    std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
    std::fs::create_dir(sidecar(&destination, ".part")).unwrap();

    let error = install_from_file(&model, &source).unwrap_err();

    assert!(
        error.contains("failed to remove"),
        "unexpected error: {error}"
    );
    assert!(!destination.exists());
}

#[test]
fn local_source_changes_cannot_replace_a_verified_destination() {
    #[derive(Clone, Copy)]
    enum Mutation {
        Grow,
        Truncate,
    }

    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let payload = (0..100_000)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    let model = test_model(&payload);
    let destination = path(&model).unwrap();
    std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
    std::fs::write(&destination, &payload).unwrap();
    let source = directory.path().join("mutable-source.onnx");

    for mutation in [Mutation::Grow, Mutation::Truncate] {
        std::fs::write(&source, &payload).unwrap();
        let changed = Cell::new(false);
        let mut progress_events = Vec::new();
        let error = install_from_file_with_progress(
            &model,
            &source,
            || false,
            |copied, total| {
                progress_events.push((copied, total));
                if copied == 0 || changed.replace(true) {
                    return;
                }
                match mutation {
                    Mutation::Grow => {
                        OpenOptions::new()
                            .append(true)
                            .open(&source)
                            .unwrap()
                            .write_all(b"x")
                            .unwrap();
                    }
                    Mutation::Truncate => {
                        OpenOptions::new()
                            .write(true)
                            .open(&source)
                            .unwrap()
                            .set_len(copied)
                            .unwrap();
                    }
                }
            },
        )
        .unwrap_err();

        assert!(error.contains("size"), "unexpected error: {error}");
        assert_manifest_progress(&progress_events, model.size_bytes);
        assert_eq!(std::fs::read(&destination).unwrap(), payload);
    }
}

#[cfg(unix)]
#[test]
fn local_import_rejects_fifo_without_blocking() {
    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let fifo = directory.path().join("model.fifo");
    create_fifo(&fifo);
    let model = test_model(b"regular model bytes");
    let started = Instant::now();

    let error = install_from_file(&model, &fifo).unwrap_err();

    assert!(
        error.contains("not a regular file"),
        "unexpected error: {error}"
    );
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[cfg(unix)]
#[test]
fn partial_and_lock_fifos_are_rejected_without_blocking() {
    let directory = tempfile::tempdir().unwrap();
    let partial = directory.path().join("model.onnx.part");
    create_fifo(&partial);
    let started = Instant::now();
    assert!(open_partial(&partial, true).is_err());
    assert!(started.elapsed() < Duration::from_secs(1));

    let metadata = directory.path().join("model.onnx.part.meta");
    create_fifo(&metadata);
    let started = Instant::now();
    let error = load_matching_metadata(&metadata, "source", "checksum", 1).unwrap_err();
    assert!(
        error.contains("not a regular file"),
        "unexpected error: {error}"
    );
    assert!(started.elapsed() < Duration::from_secs(1));

    let _serial = lock_model_environment();
    let _model_dir = ModelDirGuard::set(directory.path());
    let model = test_model(b"lock fifo test");
    let destination = path(&model).unwrap();
    let lock_path = model_lock_path(&destination).unwrap();
    std::fs::create_dir_all(lock_path.parent().unwrap()).unwrap();
    create_fifo(&lock_path);
    let started = Instant::now();
    let error = acquire_lock(&destination, &mut || false).unwrap_err();
    assert!(
        error.contains("not a regular file"),
        "unexpected error: {error}"
    );
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn remove_deletes_the_model_directory_but_keeps_the_shared_lock_namespace() {
    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let payload = b"removable model";
    let model = test_model(payload);
    let source = directory.path().join("source.onnx");
    std::fs::write(&source, payload).unwrap();
    let destination = install_from_file(&model, &source).unwrap();
    let model_directory = destination.parent().unwrap().to_path_buf();

    assert!(remove(&model).unwrap());
    assert!(!model_directory.exists());
    assert!(directory
        .path()
        .join(".locks/model-download-test.lock")
        .is_file());
    assert!(!remove(&model).unwrap());
}

#[test]
fn bearer_and_basic_credentials_are_sent_to_loopback_origin() {
    let payload = Arc::new(b"authenticated model".to_vec());
    let model = test_model(&payload);

    let bearer_payload = Arc::clone(&payload);
    let (bearer_url, bearer_server) = spawn_test_server(1, move |_, _| {
        TestResponse::ok(bearer_payload.as_ref().clone())
    });
    let bearer_directory = tempfile::tempdir().unwrap();
    let bearer_options = ModelDownloadOptions {
        authentication: Some(ModelAuthentication::Bearer("bearer-secret".into())),
        ..direct_options()
    };
    download_for_test(
        &model,
        &bearer_url,
        &bearer_options,
        bearer_directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();
    let bearer_requests = bearer_server.join().unwrap();
    assert_eq!(bearer_requests.len(), 1);
    assert_eq!(
        bearer_requests[0].header("Authorization"),
        Some("Bearer bearer-secret")
    );

    let basic_payload = Arc::clone(&payload);
    let (basic_url, basic_server) = spawn_test_server(1, move |_, _| {
        TestResponse::ok(basic_payload.as_ref().clone())
    });
    let basic_directory = tempfile::tempdir().unwrap();
    let basic_options = ModelDownloadOptions {
        authentication: Some(ModelAuthentication::Basic {
            username: "alice".into(),
            password: "s3cret".into(),
        }),
        ..direct_options()
    };
    download_for_test(
        &model,
        &basic_url,
        &basic_options,
        basic_directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();
    let basic_requests = basic_server.join().unwrap();
    assert_eq!(basic_requests.len(), 1);
    assert_eq!(
        basic_requests[0].header("Authorization"),
        Some("Basic YWxpY2U6czNjcmV0")
    );
}

#[test]
fn authenticated_url_sends_basic_auth_without_persisting_secrets() {
    let payload = Arc::new(b"authenticated URL model".to_vec());
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let (plain_url, server) = spawn_test_server(1, move |_, _| {
        TestResponse::ok(response_payload.as_ref().clone())
    });
    let authenticated_url =
        plain_url.replacen("http://", "http://url-user:p%40ss@", 1) + "?token=query-secret";
    let directory = tempfile::tempdir().unwrap();

    download_for_test(
        &model,
        &authenticated_url,
        &direct_options(),
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();

    let requests = server.join().unwrap();
    assert_eq!(
        requests[0].header("Authorization"),
        Some("Basic dXJsLXVzZXI6cEBzcw==")
    );
    assert!(requests[0].request_line.contains("token=query-secret"));
    let (_, metadata) = download_paths(directory.path());
    let metadata = std::fs::read_to_string(metadata).unwrap();
    assert!(!metadata.contains("url-user"));
    assert!(!metadata.contains("p%40ss"));
    assert!(!metadata.contains("query-secret"));
}

#[test]
fn authenticated_http_proxy_receives_the_absolute_target() {
    let payload = Arc::new(b"proxied model".to_vec());
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let (proxy_url, proxy_server) = spawn_test_server(1, move |_, _| {
        TestResponse::ok(response_payload.as_ref().clone())
    });
    let proxy_origin = Url::parse(&proxy_url).unwrap();
    let proxy = format!(
        "http://proxy-user:p%40ss@{}:{}",
        proxy_origin.host_str().unwrap(),
        proxy_origin.port().unwrap()
    );
    let options = ModelDownloadOptions {
        proxy: ModelProxy::Url(proxy),
        ..direct_options()
    };
    let directory = tempfile::tempdir().unwrap();

    download_for_test(
        &model,
        "http://models.invalid/releases/model.onnx",
        &options,
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();

    let requests = proxy_server.join().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].request_line,
        "GET http://models.invalid/releases/model.onnx HTTP/1.1"
    );
    assert_eq!(
        requests[0].header("Proxy-Authorization"),
        Some("Basic cHJveHktdXNlcjpwQHNz")
    );
}

#[test]
fn declared_full_response_size_must_match_the_manifest_before_writing() {
    let expected = b"manifest-sized model body";
    let model = test_model(expected);
    for body in [
        expected[..expected.len() - 1].to_vec(),
        [expected.as_slice(), b"x"].concat(),
    ] {
        let (source_url, server) = spawn_test_server(1, move |_, _| TestResponse::ok(body.clone()));
        let directory = tempfile::tempdir().unwrap();

        let error = download_for_test(
            &model,
            &source_url,
            &direct_options(),
            directory.path(),
            &mut || false,
            &mut |_, _| {},
        )
        .unwrap_err();

        assert!(error.contains("size mismatch"), "unexpected error: {error}");
        let (partial, metadata) = download_paths(directory.path());
        assert!(!partial.exists());
        assert!(!metadata.exists());
        assert_eq!(server.join().unwrap().len(), 1);
    }
}

#[test]
fn headerless_full_responses_are_bounded_by_the_manifest() {
    let expected = b"headerless manifest-sized model";
    let model = test_model(expected);

    let exact_body = expected.to_vec();
    let (exact_url, exact_server) = spawn_test_server(1, move |_, _| {
        TestResponse::ok(exact_body.clone()).without_content_length()
    });
    let exact_directory = tempfile::tempdir().unwrap();
    let mut exact_progress = Vec::new();
    download_for_test(
        &model,
        &exact_url,
        &direct_options(),
        exact_directory.path(),
        &mut || false,
        &mut |downloaded, total| exact_progress.push((downloaded, total)),
    )
    .unwrap();
    assert_manifest_progress(&exact_progress, model.size_bytes);
    assert_eq!(
        exact_progress.last(),
        Some(&(model.size_bytes, Some(model.size_bytes)))
    );
    assert_eq!(exact_server.join().unwrap().len(), 1);

    let short_body = expected[..expected.len() - 1].to_vec();
    let (short_url, short_server) = spawn_test_server(1, move |_, _| {
        TestResponse::ok(short_body.clone()).without_content_length()
    });
    let short_directory = tempfile::tempdir().unwrap();
    let mut short_progress = Vec::new();
    let error = download_for_test(
        &model,
        &short_url,
        &direct_options(),
        short_directory.path(),
        &mut || false,
        &mut |downloaded, total| short_progress.push((downloaded, total)),
    )
    .unwrap_err();
    assert!(error.contains("incomplete"), "unexpected error: {error}");
    assert_manifest_progress(&short_progress, model.size_bytes);
    let (short_partial, _) = download_paths(short_directory.path());
    assert_eq!(
        std::fs::metadata(short_partial).unwrap().len(),
        model.size_bytes - 1
    );
    assert_eq!(short_server.join().unwrap().len(), 1);

    let oversized_body = [expected.as_slice(), b"x"].concat();
    let (oversized_url, oversized_server) = spawn_test_server(1, move |_, _| {
        TestResponse::ok(oversized_body.clone()).without_content_length()
    });
    let oversized_directory = tempfile::tempdir().unwrap();
    let mut oversized_progress = Vec::new();
    let error = download_for_test(
        &model,
        &oversized_url,
        &direct_options(),
        oversized_directory.path(),
        &mut || false,
        &mut |downloaded, total| oversized_progress.push((downloaded, total)),
    )
    .unwrap_err();
    assert!(
        error.contains("catalog metadata"),
        "unexpected error: {error}"
    );
    assert_manifest_progress(&oversized_progress, model.size_bytes);
    let (oversized_partial, oversized_metadata) = download_paths(oversized_directory.path());
    assert!(!oversized_partial.exists());
    assert!(!oversized_metadata.exists());
    assert_eq!(oversized_server.join().unwrap().len(), 1);
}

#[test]
fn malformed_content_length_is_rejected_without_writing() {
    let payload = b"malformed-length model";
    let model = test_model(payload);
    let response_body = payload.to_vec();
    let (source_url, server) = spawn_test_server(1, move |_, _| {
        TestResponse::ok(response_body.clone()).with_raw_content_length("not-a-number")
    });
    let directory = tempfile::tempdir().unwrap();

    let error = download_for_test(
        &model,
        &source_url,
        &direct_options(),
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap_err();

    assert!(
        error.contains("invalid Content-Length"),
        "unexpected error: {error}"
    );
    let (partial, metadata) = download_paths(directory.path());
    assert!(!partial.exists());
    assert!(!metadata.exists());
    assert_eq!(server.join().unwrap().len(), 1);
}

#[test]
fn cancelled_download_resumes_with_range_and_if_range() {
    let payload = Arc::new(
        (0..200_000)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>(),
    );
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let (source_url, server) = spawn_test_server(2, move |index, request| {
        if index == 0 {
            TestResponse::ok(response_payload.as_ref().clone()).with_header("ETag", "\"resume-v1\"")
        } else {
            let start = request_range_start(request);
            let end = response_payload.len() - 1;
            TestResponse {
                status: 206,
                reason: "Partial Content",
                headers: vec![
                    (
                        "Content-Range".into(),
                        format!("bytes {start}-{end}/{}", response_payload.len()),
                    ),
                    ("ETag".into(), "\"resume-v1\"".into()),
                ],
                body: response_payload[start..].to_vec(),
                content_length: TestContentLength::Automatic,
            }
        }
    });
    let directory = tempfile::tempdir().unwrap();
    let options = direct_options();
    let cancel = Cell::new(false);
    let first_error = download_for_test(
        &model,
        &source_url,
        &options,
        directory.path(),
        &mut || cancel.get(),
        &mut |downloaded, _| {
            if downloaded > 0 {
                cancel.set(true);
            }
        },
    )
    .unwrap_err();
    assert_eq!(first_error, "cancelled");
    let (partial, _) = download_paths(directory.path());
    let interrupted_length = std::fs::metadata(&partial).unwrap().len();
    assert!(interrupted_length > 0 && interrupted_length < payload.len() as u64);

    download_for_test(
        &model,
        &source_url,
        &options,
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();

    assert_eq!(std::fs::read(partial).unwrap(), payload.as_ref().as_slice());
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[1].header("Range"),
        Some(format!("bytes={interrupted_length}-").as_str())
    );
    assert_eq!(requests[1].header("If-Range"), Some("\"resume-v1\""));
}

#[test]
fn range_ignored_with_200_truncates_the_partial() {
    let payload = Arc::new(b"complete replacement after ignored range".to_vec());
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let (source_url, server) = spawn_test_server(1, move |_, _| {
        TestResponse::ok(response_payload.as_ref().clone()).with_header("ETag", "\"v2\"")
    });
    let directory = tempfile::tempdir().unwrap();
    let prefix = &payload[..9];
    seed_partial(
        directory.path(),
        &source_url,
        &model,
        prefix,
        Some("\"v1\""),
        Some(payload.len() as u64),
    );

    download_for_test(
        &model,
        &source_url,
        &direct_options(),
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();

    let (partial, _) = download_paths(directory.path());
    assert_eq!(std::fs::read(partial).unwrap(), payload.as_ref().as_slice());
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].header("Range"), Some("bytes=9-"));
    assert_eq!(requests[0].header("If-Range"), Some("\"v1\""));
}

#[test]
fn oversized_or_exact_corrupt_partials_restart_without_a_range() {
    let payload = b"manifest bounded partial model";
    let model = test_model(payload);
    let mut corrupt = payload.to_vec();
    corrupt[0] ^= 1;
    for partial_bytes in [[payload.as_slice(), b"x"].concat(), corrupt] {
        let response_payload = payload.to_vec();
        let (source_url, server) =
            spawn_test_server(1, move |_, _| TestResponse::ok(response_payload.clone()));
        let directory = tempfile::tempdir().unwrap();
        seed_partial(
            directory.path(),
            &source_url,
            &model,
            &partial_bytes,
            None,
            Some(model.size_bytes),
        );

        download_for_test(
            &model,
            &source_url,
            &direct_options(),
            directory.path(),
            &mut || false,
            &mut |_, _| {},
        )
        .unwrap();

        let requests = server.join().unwrap();
        assert_eq!(requests[0].header("Range"), None);
    }
}

#[test]
fn range_not_satisfiable_discards_partial_then_retries_cleanly() {
    let payload = Arc::new(b"clean response after 416".to_vec());
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let (source_url, server) = spawn_test_server(2, move |index, _| {
        if index == 0 {
            TestResponse {
                status: 416,
                reason: "Range Not Satisfiable",
                headers: vec![(
                    "Content-Range".into(),
                    format!("bytes */{}", response_payload.len()),
                )],
                body: Vec::new(),
                content_length: TestContentLength::Automatic,
            }
        } else {
            TestResponse::ok(response_payload.as_ref().clone())
        }
    });
    let directory = tempfile::tempdir().unwrap();
    seed_partial(
        directory.path(),
        &source_url,
        &model,
        b"stale",
        Some("\"old\""),
        Some(payload.len() as u64),
    );

    download_for_test(
        &model,
        &source_url,
        &direct_options(),
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();

    let (partial, _) = download_paths(directory.path());
    assert_eq!(std::fs::read(partial).unwrap(), payload.as_ref().as_slice());
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].header("Range"), Some("bytes=5-"));
    assert_eq!(requests[1].header("Range"), None);
}

#[test]
fn range_not_satisfiable_completion_requires_exact_size_total_and_checksum() {
    let directory = tempfile::tempdir().unwrap();
    let partial = directory.path().join("model.part");
    let payload = b"completed ranged model";
    let model = test_model(payload);
    std::fs::write(&partial, payload).unwrap();
    let response: ureq::Response = format!(
        "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{}\r\n\r\n",
        model.size_bytes
    )
    .parse()
    .unwrap();

    assert!(handle_range_not_satisfiable(
        &response,
        &partial,
        model.size_bytes,
        model.size_bytes,
        model.sha256,
    )
    .unwrap());
    assert!(!handle_range_not_satisfiable(
        &response,
        &partial,
        model.size_bytes - 1,
        model.size_bytes,
        model.sha256,
    )
    .unwrap());
    assert!(!handle_range_not_satisfiable(
        &response,
        &partial,
        model.size_bytes,
        model.size_bytes - 1,
        model.sha256,
    )
    .unwrap());
    assert!(!handle_range_not_satisfiable(
        &response,
        &partial,
        model.size_bytes,
        model.size_bytes,
        test_sha256(b"different checksum"),
    )
    .unwrap());

    let wrong_total: ureq::Response = format!(
        "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{}\r\n\r\n",
        model.size_bytes + 1
    )
    .parse()
    .unwrap();
    assert!(!handle_range_not_satisfiable(
        &wrong_total,
        &partial,
        model.size_bytes,
        model.size_bytes,
        model.sha256,
    )
    .unwrap());
}

#[test]
fn invalid_content_range_discards_partial_then_retries_cleanly() {
    let payload = Arc::new(b"clean response after invalid content range".to_vec());
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let prefix_length = 7;
    let (source_url, server) = spawn_test_server(2, move |index, _| {
        if index == 0 {
            TestResponse {
                status: 206,
                reason: "Partial Content",
                headers: vec![(
                    "Content-Range".into(),
                    format!(
                        "bytes {}-{}/{}",
                        prefix_length + 1,
                        response_payload.len() - 1,
                        response_payload.len()
                    ),
                )],
                body: response_payload[prefix_length..].to_vec(),
                content_length: TestContentLength::Automatic,
            }
        } else {
            TestResponse::ok(response_payload.as_ref().clone())
        }
    });
    let directory = tempfile::tempdir().unwrap();
    seed_partial(
        directory.path(),
        &source_url,
        &model,
        &payload[..prefix_length],
        None,
        Some(payload.len() as u64),
    );

    download_for_test(
        &model,
        &source_url,
        &direct_options(),
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();

    let (partial, _) = download_paths(directory.path());
    assert_eq!(std::fs::read(partial).unwrap(), payload.as_ref().as_slice());
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].header("Range"), Some("bytes=7-"));
    assert_eq!(requests[1].header("Range"), None);
}

#[test]
fn bounded_wildcard_partial_response_resumes_with_manifest_progress() {
    let payload = Arc::new(b"bounded wildcard partial response".to_vec());
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let prefix_length = 8;
    let (source_url, server) = spawn_test_server(1, move |_, request| {
        let start = request_range_start(request);
        let end = response_payload.len() - 1;
        TestResponse {
            status: 206,
            reason: "Partial Content",
            headers: vec![("Content-Range".into(), format!("bytes {start}-{end}/*"))],
            body: response_payload[start..].to_vec(),
            content_length: TestContentLength::Omitted,
        }
    });
    let directory = tempfile::tempdir().unwrap();
    seed_partial(
        directory.path(),
        &source_url,
        &model,
        &payload[..prefix_length],
        None,
        None,
    );
    let mut progress = Vec::new();

    download_for_test(
        &model,
        &source_url,
        &direct_options(),
        directory.path(),
        &mut || false,
        &mut |downloaded, total| progress.push((downloaded, total)),
    )
    .unwrap();

    assert_manifest_progress(&progress, model.size_bytes);
    let requests = server.join().unwrap();
    assert_eq!(requests[0].header("Range"), Some("bytes=8-"));
}

#[test]
fn invalid_partial_sizes_and_lengths_reset_before_a_clean_retry() {
    #[derive(Clone, Copy)]
    enum InvalidPartial {
        SmallerTotal,
        LargerTotal,
        RangePastManifest,
        WrongLength,
        MalformedLength,
    }

    for invalid in [
        InvalidPartial::SmallerTotal,
        InvalidPartial::LargerTotal,
        InvalidPartial::RangePastManifest,
        InvalidPartial::WrongLength,
        InvalidPartial::MalformedLength,
    ] {
        let payload = Arc::new(b"partial response bounds model".to_vec());
        let model = test_model(&payload);
        let response_payload = Arc::clone(&payload);
        let prefix_length = 6;
        let (source_url, server) = spawn_test_server(2, move |index, _| {
            if index != 0 {
                return TestResponse::ok(response_payload.as_ref().clone());
            }
            let size = response_payload.len();
            let (end, total, body) = match invalid {
                InvalidPartial::SmallerTotal => (
                    size - 2,
                    (size - 1).to_string(),
                    response_payload[prefix_length..size - 1].to_vec(),
                ),
                InvalidPartial::LargerTotal => (
                    size - 1,
                    (size + 1).to_string(),
                    response_payload[prefix_length..].to_vec(),
                ),
                InvalidPartial::RangePastManifest => {
                    let mut body = response_payload[prefix_length..].to_vec();
                    body.push(0);
                    (size, "*".to_string(), body)
                }
                InvalidPartial::WrongLength | InvalidPartial::MalformedLength => (
                    size - 1,
                    size.to_string(),
                    response_payload[prefix_length..].to_vec(),
                ),
            };
            let response = TestResponse {
                status: 206,
                reason: "Partial Content",
                headers: vec![(
                    "Content-Range".into(),
                    format!("bytes {prefix_length}-{end}/{total}"),
                )],
                body,
                content_length: TestContentLength::Automatic,
            };
            match invalid {
                InvalidPartial::WrongLength => response.with_raw_content_length("1"),
                InvalidPartial::MalformedLength => response.with_raw_content_length("not-a-number"),
                _ => response,
            }
        });
        let directory = tempfile::tempdir().unwrap();
        seed_partial(
            directory.path(),
            &source_url,
            &model,
            &payload[..prefix_length],
            None,
            Some(model.size_bytes),
        );

        download_for_test(
            &model,
            &source_url,
            &direct_options(),
            directory.path(),
            &mut || false,
            &mut |_, _| {},
        )
        .unwrap();

        let requests = server.join().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].header("Range"), Some("bytes=6-"));
        assert_eq!(requests[1].header("Range"), None);
    }
}

#[test]
fn changed_source_identity_does_not_send_range() {
    let payload = Arc::new(b"model from a different source".to_vec());
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let (source_url, server) = spawn_test_server(1, move |_, _| {
        TestResponse::ok(response_payload.as_ref().clone())
    });
    let directory = tempfile::tempdir().unwrap();
    seed_partial(
        directory.path(),
        "http://old-source.invalid/model.onnx",
        &model,
        b"old partial",
        Some("\"old\""),
        Some(payload.len() as u64),
    );

    download_for_test(
        &model,
        &source_url,
        &direct_options(),
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();

    let (partial, _) = download_paths(directory.path());
    assert_eq!(std::fs::read(partial).unwrap(), payload.as_ref().as_slice());
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].header("Range"), None);
    assert_eq!(requests[0].header("If-Range"), None);
}

#[test]
fn rotated_signed_url_resumes_the_same_verified_object() {
    let payload = Arc::new(b"signed URL resume payload".to_vec());
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let (base_url, server) = spawn_test_server(1, move |_, request| {
        let start = request_range_start(request);
        let end = response_payload.len() - 1;
        TestResponse {
            status: 206,
            reason: "Partial Content",
            headers: vec![
                (
                    "Content-Range".into(),
                    format!("bytes {start}-{end}/{}", response_payload.len()),
                ),
                ("ETag".into(), "\"stable-object\"".into()),
            ],
            body: response_payload[start..].to_vec(),
            content_length: TestContentLength::Automatic,
        }
    });
    let old_url = format!("{base_url}?signature=expired");
    let new_url = format!("{base_url}?signature=fresh");
    let directory = tempfile::tempdir().unwrap();
    let prefix_length = 7;
    seed_partial(
        directory.path(),
        &old_url,
        &model,
        &payload[..prefix_length],
        Some("\"stable-object\""),
        Some(payload.len() as u64),
    );

    download_for_test(
        &model,
        &new_url,
        &direct_options(),
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();

    let requests = server.join().unwrap();
    assert_eq!(requests[0].header("Range"), Some("bytes=7-"));
    assert_eq!(requests[0].header("If-Range"), Some("\"stable-object\""));
    assert!(requests[0].request_line.contains("signature=fresh"));
}

#[test]
fn legacy_v1_metadata_resumes_with_missing_null_or_matching_total() {
    #[derive(Clone, Copy)]
    enum LegacyTotal {
        Missing,
        Null,
        Matching,
    }

    for legacy_total in [
        LegacyTotal::Missing,
        LegacyTotal::Null,
        LegacyTotal::Matching,
    ] {
        let payload = Arc::new(b"legacy metadata resume model".to_vec());
        let model = test_model(&payload);
        let response_payload = Arc::clone(&payload);
        let prefix_length = 7;
        let (source_url, server) = spawn_test_server(1, move |_, request| {
            let start = request_range_start(request);
            let end = response_payload.len() - 1;
            TestResponse {
                status: 206,
                reason: "Partial Content",
                headers: vec![(
                    "Content-Range".into(),
                    format!("bytes {start}-{end}/{}", response_payload.len()),
                )],
                body: response_payload[start..].to_vec(),
                content_length: TestContentLength::Automatic,
            }
        });
        let directory = tempfile::tempdir().unwrap();
        let (partial, metadata_path) = download_paths(directory.path());
        std::fs::write(&partial, &payload[..prefix_length]).unwrap();
        let mut metadata = serde_json::json!({
            "version": 1,
            "source_id": source_identity(&source_url),
            "expected_sha256": model.sha256,
            "etag": null,
            "last_modified": null,
            "total": null,
        });
        match legacy_total {
            LegacyTotal::Missing => {
                metadata.as_object_mut().unwrap().remove("total");
            }
            LegacyTotal::Null => {}
            LegacyTotal::Matching => {
                metadata["total"] = serde_json::json!(model.size_bytes);
            }
        }
        std::fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();

        download_for_test(
            &model,
            &source_url,
            &direct_options(),
            directory.path(),
            &mut || false,
            &mut |_, _| {},
        )
        .unwrap();

        assert_eq!(std::fs::read(partial).unwrap(), payload.as_ref().as_slice());
        let requests = server.join().unwrap();
        assert_eq!(requests[0].header("Range"), Some("bytes=7-"));
    }
}

#[test]
fn conflicting_v1_metadata_total_resets_the_partial() {
    let payload = Arc::new(b"conflicting legacy metadata model".to_vec());
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let (source_url, server) = spawn_test_server(1, move |_, _| {
        TestResponse::ok(response_payload.as_ref().clone())
    });
    let directory = tempfile::tempdir().unwrap();
    let (partial, metadata_path) = download_paths(directory.path());
    std::fs::write(&partial, &payload[..6]).unwrap();
    let metadata = serde_json::json!({
        "version": 1,
        "source_id": source_identity(&source_url),
        "expected_sha256": model.sha256,
        "etag": null,
        "last_modified": null,
        "total": model.size_bytes + 1,
    });
    std::fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();

    download_for_test(
        &model,
        &source_url,
        &direct_options(),
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();

    assert_eq!(std::fs::read(partial).unwrap(), payload.as_ref().as_slice());
    let requests = server.join().unwrap();
    assert_eq!(requests[0].header("Range"), None);
}

#[test]
fn checksum_mismatch_discards_partial_and_metadata() {
    let expected = b"expected model bytes";
    let model = test_model(expected);
    let (source_url, server) =
        spawn_test_server(1, |_, _| TestResponse::ok(b"tampered model bytes".to_vec()));
    let directory = tempfile::tempdir().unwrap();

    let error = download_for_test(
        &model,
        &source_url,
        &direct_options(),
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap_err();

    assert!(
        error.contains("checksum mismatch"),
        "unexpected error: {error}"
    );
    let (partial, metadata) = download_paths(directory.path());
    assert!(!partial.exists());
    assert!(!metadata.exists());
    assert_eq!(server.join().unwrap().len(), 1);
}

#[test]
fn failed_update_keeps_the_existing_verified_model() {
    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let payload = b"currently installed verified model";
    let model = test_model(payload);
    let destination = path(&model).unwrap();
    std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
    std::fs::write(&destination, payload).unwrap();
    let (source_url, server) = spawn_test_server(1, |_, _| TestResponse {
        status: 503,
        reason: "Service Unavailable",
        headers: Vec::new(),
        body: Vec::new(),
        content_length: TestContentLength::Automatic,
    });
    let options = ModelDownloadOptions {
        source_url: Some(source_url),
        ..direct_options()
    };

    let error = update_with_options(&model, &options).unwrap_err();

    assert!(error.contains("HTTP 503"), "unexpected error: {error}");
    assert_eq!(std::fs::read(&destination).unwrap(), payload);
    assert_eq!(verify_at(&model, &destination).unwrap(), destination);
    assert_eq!(server.join().unwrap().len(), 1);
}

#[test]
fn wrong_sized_alternate_and_redirected_updates_preserve_the_installed_model() {
    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let payload = b"verified model preserved across bad mirrors";
    let model = test_model(payload);
    let destination = path(&model).unwrap();
    std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
    std::fs::write(&destination, payload).unwrap();

    let oversized = [payload.as_slice(), b"x"].concat();
    let (alternate_url, alternate_server) =
        spawn_test_server(1, move |_, _| TestResponse::ok(oversized.clone()));
    let alternate_error = update_with_options(
        &model,
        &ModelDownloadOptions {
            source_url: Some(alternate_url),
            ..direct_options()
        },
    )
    .unwrap_err();
    assert!(
        alternate_error.contains("size mismatch"),
        "unexpected error: {alternate_error}"
    );
    assert_eq!(std::fs::read(&destination).unwrap(), payload);
    assert_eq!(alternate_server.join().unwrap().len(), 1);

    let short = payload[..payload.len() - 1].to_vec();
    let (redirect_url, redirect_server) = spawn_test_server(2, move |index, _| {
        if index == 0 {
            TestResponse {
                status: 302,
                reason: "Found",
                headers: vec![("Location".into(), "/redirected/model.onnx".into())],
                body: Vec::new(),
                content_length: TestContentLength::Automatic,
            }
        } else {
            TestResponse::ok(short.clone())
        }
    });
    let redirect_error = update_with_options(
        &model,
        &ModelDownloadOptions {
            source_url: Some(redirect_url),
            ..direct_options()
        },
    )
    .unwrap_err();
    assert!(
        redirect_error.contains("size mismatch"),
        "unexpected error: {redirect_error}"
    );
    assert_eq!(std::fs::read(&destination).unwrap(), payload);
    assert_eq!(redirect_server.join().unwrap().len(), 2);
}

#[test]
fn offline_update_keeps_an_existing_verified_model() {
    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let payload = b"verified model available offline";
    let model = test_model(payload);
    let destination = path(&model).unwrap();
    std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
    std::fs::write(&destination, payload).unwrap();
    let options = ModelDownloadOptions {
        offline: true,
        ..direct_options()
    };

    assert_eq!(update_with_options(&model, &options).unwrap(), destination);
    assert_eq!(std::fs::read(destination).unwrap(), payload);
}

#[test]
fn concurrent_installs_share_one_download() {
    let _serial = lock_model_environment();
    let directory = tempfile::tempdir().unwrap();
    let _model_dir = ModelDirGuard::set(directory.path());
    let payload = Arc::new(
        (0..100_000)
            .map(|index| (index % 239) as u8)
            .collect::<Vec<_>>(),
    );
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let (source_url, server) = spawn_test_server(1, move |_, _| {
        std::thread::sleep(Duration::from_millis(100));
        TestResponse::ok(response_payload.as_ref().clone())
    });
    let options = ModelDownloadOptions {
        source_url: Some(source_url),
        ..direct_options()
    };
    let barrier = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();
    for _ in 0..2 {
        let worker_barrier = Arc::clone(&barrier);
        let worker_options = options.clone();
        workers.push(std::thread::spawn(move || {
            worker_barrier.wait();
            install_with_options(&model, &worker_options)
        }));
    }
    barrier.wait();
    let first = workers.remove(0).join().unwrap().unwrap();
    let second = workers.remove(0).join().unwrap().unwrap();

    assert_eq!(first, second);
    assert_eq!(std::fs::read(first).unwrap(), payload.as_ref().as_slice());
    assert_eq!(server.join().unwrap().len(), 1);
}

#[test]
fn same_host_redirect_preserves_bearer_authentication() {
    let payload = Arc::new(b"same-host redirect model".to_vec());
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let (source_url, server) = spawn_test_server(2, move |index, _| {
        if index == 0 {
            TestResponse {
                status: 302,
                reason: "Found",
                headers: vec![("Location".into(), "/redirected/model.onnx".into())],
                body: Vec::new(),
                content_length: TestContentLength::Automatic,
            }
        } else {
            TestResponse::ok(response_payload.as_ref().clone())
        }
    });
    let directory = tempfile::tempdir().unwrap();
    let options = ModelDownloadOptions {
        authentication: Some(ModelAuthentication::Bearer("same-host-secret".into())),
        ..direct_options()
    };

    download_for_test(
        &model,
        &source_url,
        &options,
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();

    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].header("Authorization"),
        Some("Bearer same-host-secret")
    );
    assert_eq!(
        requests[1].header("Authorization"),
        Some("Bearer same-host-secret")
    );
    assert!(requests[1]
        .request_line
        .starts_with("GET /redirected/model.onnx "));
}

#[test]
fn cross_origin_redirect_does_not_forward_bearer_authentication() {
    let payload = Arc::new(b"cross-origin redirect model".to_vec());
    let model = test_model(&payload);
    let response_payload = Arc::clone(&payload);
    let (destination_url, destination_server) = spawn_test_server(1, move |_, _| {
        TestResponse::ok(response_payload.as_ref().clone())
    });
    let mut destination = Url::parse(&destination_url).unwrap();
    destination.set_host(Some("localhost")).unwrap();
    destination.set_path("/redirect-target/model.onnx");
    let redirected_to = destination.to_string();
    let (source_url, source_server) = spawn_test_server(1, move |_, _| TestResponse {
        status: 302,
        reason: "Found",
        headers: vec![("Location".into(), redirected_to.clone())],
        body: Vec::new(),
        content_length: TestContentLength::Automatic,
    });
    let directory = tempfile::tempdir().unwrap();
    let options = ModelDownloadOptions {
        authentication: Some(ModelAuthentication::Bearer("origin-only-secret".into())),
        ..direct_options()
    };

    download_for_test(
        &model,
        &source_url,
        &options,
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap();

    let source_requests = source_server.join().unwrap();
    let destination_requests = destination_server.join().unwrap();
    assert_eq!(source_requests.len(), 1);
    assert_eq!(destination_requests.len(), 1);
    assert_eq!(
        source_requests[0].header("Authorization"),
        Some("Bearer origin-only-secret")
    );
    assert_eq!(destination_requests[0].header("Authorization"), None);
}

#[test]
fn redirect_to_a_signed_http_url_is_rejected_before_the_second_request() {
    let model = test_model(b"redirected signed model");
    let (source_url, server) = spawn_test_server(1, |_, _| TestResponse {
        status: 302,
        reason: "Found",
        headers: vec![(
            "Location".into(),
            "http://models.invalid/model.onnx?signature=redirect-secret".into(),
        )],
        body: Vec::new(),
        content_length: TestContentLength::Automatic,
    });
    let directory = tempfile::tempdir().unwrap();

    let error = download_for_test(
        &model,
        &source_url,
        &direct_options(),
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap_err();

    assert!(
        error.contains("non-HTTPS transport"),
        "unexpected error: {error}"
    );
    assert!(!error.contains("redirect-secret"));
    assert_eq!(server.join().unwrap().len(), 1);
}

#[test]
fn transport_error_redacts_url_userinfo_and_query() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let source_url = format!(
        "http://url-user:url-password@{}/model.onnx?token=query-secret#fragment",
        listener.local_addr().unwrap()
    );
    let model = test_model(b"unreachable model");
    let options = ModelDownloadOptions {
        connect_timeout: Duration::from_millis(100),
        read_timeout: Duration::from_millis(100),
        request_timeout: Duration::from_millis(300),
        ..direct_options()
    };
    let directory = tempfile::tempdir().unwrap();

    let error = download_for_test(
        &model,
        &source_url,
        &options,
        directory.path(),
        &mut || false,
        &mut |_, _| {},
    )
    .unwrap_err();

    assert!(error.contains("model download from http://127.0.0.1:"));
    assert!(!error.contains("url-user"), "username leaked: {error}");
    assert!(!error.contains("url-password"), "password leaked: {error}");
    assert!(!error.contains("query-secret"), "query leaked: {error}");
    assert!(!error.contains("token="), "query key leaked: {error}");
}

#[test]
fn environment_proxy_resolution_honors_no_proxy() {
    let _serial = lock_model_environment();
    let _http_proxy = EnvVarGuard::set("HTTP_PROXY", Some("http://127.0.0.1:18080"));
    let _http_proxy_lower = EnvVarGuard::set("http_proxy", None);
    let _all_proxy = EnvVarGuard::set("ALL_PROXY", None);
    let _all_proxy_lower = EnvVarGuard::set("all_proxy", None);
    let _no_proxy_lower = EnvVarGuard::set("no_proxy", None);
    let _no_proxy = EnvVarGuard::set("NO_PROXY", Some("127.0.0.1"));
    let source = Url::parse("http://127.0.0.1:8080/model.onnx").unwrap();

    assert!(resolve_proxy(&source, &ModelProxy::Environment)
        .unwrap()
        .is_none());

    std::env::set_var("NO_PROXY", "example.invalid");
    assert!(resolve_proxy(&source, &ModelProxy::Environment)
        .unwrap()
        .is_some());
}

#[test]
fn proxy_urls_validate_ports_and_normalize_scheme_and_credentials() {
    let source = Url::parse("http://models.example/model.onnx").unwrap();
    let invalid = ModelProxy::Url("http://user:secret@proxy.example:notaport".into());
    let error = resolve_proxy(&source, &invalid).unwrap_err();
    assert!(
        error.contains("value redacted"),
        "unexpected error: {error}"
    );
    assert!(!error.contains("secret"));

    let (normalized, authorization) =
        normalize_proxy_url("HTTP://user:p%40ss@proxy.example:8080/").unwrap();
    assert_eq!(normalized, "http://user:p@ss@proxy.example:8080");
    assert_eq!(authorization.as_deref(), Some("Basic dXNlcjpwQHNz"));
    assert!(normalize_proxy_url("http://proxy.example:8080/path").is_err());
    assert!(normalize_proxy_url("http://[::1]:8080").is_err());
}
