import SwiftUI

/// Generic scrollable text view used by the Legal section to surface the
/// privacy policy, ToS, disclaimer, and OSS acknowledgement bodies.
struct LegalDocumentView: View {

    let title: String
    let documentBody: String

    init(title: String, body: String) {
        self.title = title
        self.documentBody = body
    }

    var body: some View {
        ScrollView {
            Text(documentBody)
                .font(.body)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.leading, 20)
                .padding(.trailing, 20)
                .padding(.top, 16)
                .padding(.bottom, 16)
                .textSelection(.enabled)
        }
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
    }
}
