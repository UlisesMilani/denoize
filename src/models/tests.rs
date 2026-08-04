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
}

impl TestResponse {
    fn ok(body: impl Into<Vec<u8>>) -> Self {
        Self {
            status: 200,
            reason: "OK",
            headers: Vec::new(),
            body: body.into(),
        }
    }

    fn with_header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers.push((name.to_string(), value.into()));
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
        license: "MIT",
        sample_rate: 16_000,
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
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        response.status,
        response.reason,
        response.body.len()
    );
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

#[test]
fn manifest_has_pinned_integrity_and_metadata() {
    for model in MODELS {
        assert_eq!(model.sha256.len(), 64);
        assert_eq!(model.revision.len(), 40);
        assert!(model.url.contains(model.revision));
        assert!(model.sample_rate > 0);
        assert!(!model.license.is_empty());
    }
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
    assert!(load_matching_metadata(&metadata, "source", "checksum")
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
    let model = test_model(payload);
    let destination = path(&model).unwrap();
    std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
    std::fs::write(sidecar(&destination, ".part"), payload).unwrap();
    let options = ModelDownloadOptions {
        offline: true,
        ..direct_options()
    };

    assert_eq!(install_with_options(&model, &options).unwrap(), destination);
    assert_eq!(std::fs::read(&destination).unwrap(), payload);
    assert!(!sidecar(&destination, ".part").exists());
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
    std::fs::write(&valid_source, expected).unwrap();
    std::fs::write(&invalid_source, b"tampered local model").unwrap();

    let installed = install_from_file(&model, &valid_source).unwrap();
    assert_eq!(std::fs::read(&installed).unwrap(), expected);

    let error = install_from_file(&model, &invalid_source).unwrap_err();
    assert!(
        error.contains("checksum mismatch"),
        "unexpected error: {error}"
    );
    assert_eq!(std::fs::read(installed).unwrap(), expected);
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
    let error = load_matching_metadata(&metadata, "source", "checksum").unwrap_err();
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
fn checksum_mismatch_discards_partial_and_metadata() {
    let expected = b"expected model bytes";
    let model = test_model(expected);
    let (source_url, server) =
        spawn_test_server(1, |_, _| TestResponse::ok(b"corrupt model bytes".to_vec()));
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
