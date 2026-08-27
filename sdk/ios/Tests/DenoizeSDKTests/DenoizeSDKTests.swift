import XCTest
@testable import DenoizeSDK

final class DenoizeSDKTests: XCTestCase {
    func testVersionAndFrozenLifecycleNames() throws {
        XCTAssertFalse(try DenoizeProcessor.version().isEmpty)
        XCTAssertEqual(MobileSessionState.rebuildRequired.rawValue, "rebuild-required")
    }

    func testLifecycleRejectsStaleAndOutOfOrderTransitions() throws {
        let route = try AudioRoute(sampleRate: 48_000, bufferFrames: 256, channels: 1)
        let session = try DenoizeMobileSession()
        XCTAssertThrowsError(try session.start())
        try session.configure(route: route)
        XCTAssertEqual(session.routeGeneration, 1)
        XCTAssertThrowsError(try session.configure(route: route))
        try session.start()
        try session.interrupted()
        XCTAssertNil(session.route)
        try session.resume(route: route)
        XCTAssertEqual(session.routeGeneration, 2)
        try session.close()
        XCTAssertThrowsError(try session.resume(route: route))
    }
}
