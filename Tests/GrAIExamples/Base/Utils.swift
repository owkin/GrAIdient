//
// Utils.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 19/10/2022.
//

import Foundation
import Cocoa
import GrAIdient

/// Python library default path.
let PYTHON_LIB =
    FileManager.default.homeDirectoryForCurrentUser.path +
    "/miniconda3/envs/graiexamples/lib/libpython3.9.dylib"

/// Set the Python library path.
func setPythonLib()
{
    setenv("PYTHON_LIBRARY", PYTHON_LIB, 1)
}
