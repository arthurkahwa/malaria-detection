import SwiftUI

/// Top-level tab shell. Per spec §11, four tabs are always visible:
/// Home, History, Settings, About. Home and History are gated by
/// `AuthGate` once Phase 6 lands; Settings and About are always
/// accessible.
struct RootView: View {
    var body: some View {
        TabView {
            HomeTab()
                .tabItem {
                    Label("Home", systemImage: "viewfinder")
                }

            HistoryTab()
                .tabItem {
                    Label("History", systemImage: "clock.arrow.circlepath")
                }

            SettingsTab()
                .tabItem {
                    Label("Settings", systemImage: "gearshape")
                }

            AboutTab()
                .tabItem {
                    Label("About", systemImage: "info.circle")
                }
        }
    }
}
