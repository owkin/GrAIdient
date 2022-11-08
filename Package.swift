// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required
// to build this package.

import PackageDescription

let package = Package(
    name: "MAKit",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        .library(
            name: "MAKit",
            targets: ["MAKit", "MATestsUtils"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/pvieito/PythonKit",
            branch: "master"
        ),
    ],
    targets: [
        .target(
            name: "MAKit",
            dependencies: [],
            resources: [
                .copy("Metal/Kernel")
            ]
        ),
        .target(
            name: "MATestsUtils",
            dependencies: ["MAKit"]
        ),
        .testTarget(
            name: "MAKitTests",
            dependencies: ["MAKit", "MATestsUtils"]
        ),
        .testTarget(
            name: "MATorchTests",
            dependencies: ["MAKit", "PythonKit"],
            resources: [
                .copy("Base/python_lib"),
                .copy("Base/setup.py")
            ]
        ),
        .testTarget(
            name: "MAExamples",
            dependencies: ["MAKit", "PythonKit"],
            resources: [
                .copy("Base/data"),
                .copy("Base/python_lib"),
                .copy("Base/setup.py")
            ]
        ),
    ]
)
