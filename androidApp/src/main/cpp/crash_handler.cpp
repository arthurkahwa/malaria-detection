#include <jni.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

// -----------------------------------------------------------------------
// Global state — must be async-signal-safe (no heap, no JVM references).
// -----------------------------------------------------------------------

// Pre-configured crash directory, set once from the JVM thread in install().
static char g_crash_dir[512];

// Previous signal handlers restored by uninstall().
static struct sigaction g_prev_sigsegv;
static struct sigaction g_prev_sigbus;
static struct sigaction g_prev_sigfpe;
static struct sigaction g_prev_sigill;
static struct sigaction g_prev_sigabrt;

static volatile int g_installed = 0;

// -----------------------------------------------------------------------
// Signal handler — async-signal-safe POSIX only (open/write/close/snprintf).
// -----------------------------------------------------------------------

static void crash_signal_handler(int signum, siginfo_t* /*info*/, void* /*ctx*/) {
    if (g_crash_dir[0] != '\0') {
        // Build filename: <crashDir>/native_crash_<signum>.json
        char filepath[600];
        snprintf(filepath, sizeof(filepath), "%s/native_crash_%d.json", g_crash_dir, signum);

        // Build JSON content on the stack — no malloc.
        char content[256];
        int len = snprintf(content, sizeof(content),
            "{\"native_crash\":true,\"signal\":%d,\"timestamp\":\"(not available in signal context)\"}",
            signum);

        int fd = open(filepath, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd >= 0) {
            write(fd, content, (size_t)(len > 0 ? len : 0));
            close(fd);
        }
    }

    // Re-raise with the previous handler so the OS still gets its crash report.
    struct sigaction* prev = nullptr;
    switch (signum) {
        case SIGSEGV: prev = &g_prev_sigsegv; break;
        case SIGBUS:  prev = &g_prev_sigbus;  break;
        case SIGFPE:  prev = &g_prev_sigfpe;  break;
        case SIGILL:  prev = &g_prev_sigill;  break;
        case SIGABRT: prev = &g_prev_sigabrt; break;
        default: break;
    }

    if (prev != nullptr) {
        sigaction(signum, prev, nullptr);
    } else {
        // Restore default so the process actually terminates.
        struct sigaction sa_default;
        sigemptyset(&sa_default.sa_mask);
        sa_default.sa_flags = 0;
        sa_default.sa_handler = SIG_DFL;
        sigaction(signum, &sa_default, nullptr);
    }

    // Re-raise the signal to let the OS handle it normally.
    raise(signum);
}

// -----------------------------------------------------------------------
// JNI — runs on the JVM thread; JNI calls are allowed here.
// -----------------------------------------------------------------------

extern "C" JNIEXPORT void JNICALL
Java_com_malaria_android_services_NativeCrashHandler_install(
        JNIEnv* env, jobject /*thiz*/, jstring crashDir) {

    if (crashDir == nullptr) return;

    const char* dir = env->GetStringUTFChars(crashDir, nullptr);
    if (dir == nullptr) return;

    // Copy into the global char array so the signal handler can use it safely.
    strncpy(g_crash_dir, dir, sizeof(g_crash_dir) - 1);
    g_crash_dir[sizeof(g_crash_dir) - 1] = '\0';

    env->ReleaseStringUTFChars(crashDir, dir);

    // Register signal handlers, saving the previous ones.
    struct sigaction sa;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sa.sa_sigaction = crash_signal_handler;

    sigaction(SIGSEGV, &sa, &g_prev_sigsegv);
    sigaction(SIGBUS,  &sa, &g_prev_sigbus);
    sigaction(SIGFPE,  &sa, &g_prev_sigfpe);
    sigaction(SIGILL,  &sa, &g_prev_sigill);
    sigaction(SIGABRT, &sa, &g_prev_sigabrt);

    g_installed = 1;
}

extern "C" JNIEXPORT void JNICALL
Java_com_malaria_android_services_NativeCrashHandler_uninstall(
        JNIEnv* /*env*/, jobject /*thiz*/) {

    if (!g_installed) return;

    // Restore the original handlers.
    sigaction(SIGSEGV, &g_prev_sigsegv, nullptr);
    sigaction(SIGBUS,  &g_prev_sigbus,  nullptr);
    sigaction(SIGFPE,  &g_prev_sigfpe,  nullptr);
    sigaction(SIGILL,  &g_prev_sigill,  nullptr);
    sigaction(SIGABRT, &g_prev_sigabrt, nullptr);

    g_crash_dir[0] = '\0';
    g_installed = 0;
}
