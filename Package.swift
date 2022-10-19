// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MAKit",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
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
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
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
                .process("Base/python_lib"),
                .process("Base/setup.py")
            ]
        ),
        .testTarget(
            name: "MAExamples",
            dependencies: ["MAKit", "PythonKit"],
            resources: [
                .process("Base/data"),
                .process("Base/python_lib"),
                .process("Base/setup.py")
            ]
        ),
    ]
)
