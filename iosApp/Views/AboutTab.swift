import SwiftUI

/// About tab (spec §11). Static content: app name + version, build
/// number, Hippocratic License link, source code link, maintainer credit.
struct AboutTab: View {

    private var appName: String {
        Bundle.main.infoDictionary?["CFBundleName"] as? String ?? "Malaria Detector"
    }

    private var version: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.1.0"
    }

    private var build: String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(appName)
                            .font(.title2)
                            .fontWeight(.semibold)
                        Text("Version \(version) (build \(build))")
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 6)
                    .padding(.bottom, 6)
                }

                Section("License") {
                    Link(destination: URL(string: AboutLinks.hippocraticLicense)!) {
                        HStack {
                            Text("Hippocratic License 3.0")
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section("Source") {
                    Link(destination: URL(string: AboutLinks.sourceCode)!) {
                        HStack {
                            Text("Source code")
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                                .foregroundStyle(.secondary)
                        }
                    }
                    Link(destination: URL(string: AboutLinks.sourceCode)!) {
                        HStack {
                            Text("Contributors")
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section("Maintainer") {
                    Text(AboutLinks.maintainer)
                }
            }
            .navigationTitle("About")
        }
    }
}

/// Centralised constants so the Android side stays in lockstep value-for-value.
enum AboutLinks {
    static let hippocraticLicense = "https://firstdonoharm.dev/version/3/0/"
    static let sourceCode = "https://github.com/arthurkahwa/malaria-detection"
    static let maintainer = "Arthur Kahwa"
}
