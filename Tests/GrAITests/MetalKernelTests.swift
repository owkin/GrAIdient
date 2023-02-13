import XCTest
import GrAIdient

/// Test that the different GPU kernel are valid.
final class MetalKernelTests: XCTestCase
{
    func testGet() throws
    {
        _ = MetalKernel.get
    }
}
