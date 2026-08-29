// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "DenoizeSDK",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
    ],
    products: [
        .library(name: "DenoizeSDK", targets: ["DenoizeSDK"]),
    ],
    targets: [
        .binaryTarget(
            name: "CDenoize",
            path: "DenoizeC.xcframework"
        ),
        .target(
            name: "DenoizeSDK",
            dependencies: ["CDenoize"],
            path: "Sources/DenoizeSDK"
        ),
        .testTarget(
            name: "DenoizeSDKTests",
            dependencies: ["DenoizeSDK"],
            path: "Tests/DenoizeSDKTests"
        ),
    ]
)
