fn main() {
    match denoize_desktop_lib::accessibility_e2e_requested_from_args() {
        Ok(true) => {
            denoize_desktop_lib::run_accessibility_e2e();
            return;
        }
        Ok(false) => {}
        Err(error) => {
            eprintln!("denoize desktop accessibility E2E: {error}");
            std::process::exit(2);
        }
    }
    match denoize_desktop_lib::job_worker_request_from_args() {
        Ok(Some(request)) => {
            std::process::exit(denoize_desktop_lib::run_job_worker(&request));
        }
        Ok(None) => {}
        Err(error) => {
            eprintln!("denoize desktop job worker: {error}");
            std::process::exit(2);
        }
    }
    match denoize_desktop_lib::preview_worker_request_from_args() {
        Ok(Some(request)) => {
            std::process::exit(denoize_desktop_lib::run_preview_worker(&request));
        }
        Ok(None) => {}
        Err(error) => {
            eprintln!("denoize preview worker: {error}");
            std::process::exit(2);
        }
    }
    denoize_desktop_lib::run();
}
