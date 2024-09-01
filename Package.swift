// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required
// to build this package.

import PackageDescription

let package = Package(
    name: "GrAIdient",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "GrAIdient",
            targets: ["GrAIdient", "GrAITestsUtils"]
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
            name: "GrAIdient",
            dependencies: [],
            resources: [
                .copy("Metal/Kernel")
            ]
        ),
        .target(
            name: "GrAITestsUtils",
            dependencies: ["GrAIdient"]
        ),
        .testTarget(
            name: "GrAITests",
            dependencies: ["GrAIdient", "GrAITestsUtils"]
        ),
        .testTarget(
            name: "GrAITorchTests",
            dependencies: ["GrAIdient", "PythonKit"],
            resources: [
                .copy("Base/python_lib"),
                .copy("Base/setup.py")
            ]
        ),
        .testTarget(
            name: "GrAIExamples",
            dependencies: ["GrAIdient", "PythonKit"],
            resources: [
                .copy("Base/python_lib"),
                .copy("Base/setup.py")
            ]
        ),
    ]
)
