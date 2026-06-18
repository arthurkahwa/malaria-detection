package com.malaria.android.services

import java.io.File

object NativeCrashHandler {
    init {
        System.loadLibrary("malaria_crash_handler")
    }
    external fun install(crashDir: String)
    external fun uninstall()
}
