import SwiftUI

/// Legal section (spec §11). Privacy + ToS link to external URLs (v1
/// uses a placeholder note). Decision-support disclaimer + open-source
/// acknowledgements open in-app readers.
struct LegalSection: View {

    var body: some View {
        Section {
            NavigationLink {
                LegalDocumentView(
                    title: "Privacy policy",
                    body: "The full privacy policy is published at the project's source-code repository. v1 ships without an in-app copy because the wording is finalised by each deploying clinic."
                )
            } label: {
                Text("Privacy policy")
            }
            NavigationLink {
                LegalDocumentView(
                    title: "Terms of service",
                    body: "The terms of service are governed by the Hippocratic License 3.0 (HL3-FULL). The full text is shown in the Decision-support disclaimer below and in the About tab."
                )
            } label: {
                Text("Terms of service")
            }
            NavigationLink {
                LegalDocumentView(
                    title: "Decision-support disclaimer",
                    body: LegalText.disclaimerBody
                )
            } label: {
                Text("Decision-support disclaimer")
            }
            NavigationLink {
                LegalDocumentView(
                    title: "Open-source acknowledgements",
                    body: "This software is built on open-source components including SwiftUI, SwiftData, Core ML, Kotlin Multiplatform, Compose, Room, SQLCipher, and the Hippocratic License. Full attribution travels with each release."
                )
            } label: {
                Text("Open-source acknowledgements")
            }
        } header: {
            Text("Legal")
        }
    }
}
