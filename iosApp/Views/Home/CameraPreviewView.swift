import AVFoundation
import SwiftUI
import UIKit

/// `UIViewRepresentable` wrapping `AVCaptureVideoPreviewLayer`. The
/// preview layer is attached as the host view's backing layer so it
/// resizes with autolayout. Per spec §11 the camera is portrait-only,
/// so we pin the connection's rotation angle to 90° (portrait).
struct CameraPreviewView: UIViewRepresentable {

    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewContainerView {
        let view = PreviewContainerView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        if let connection = view.previewLayer.connection,
           connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
        return view
    }

    func updateUIView(_ uiView: PreviewContainerView, context: Context) {
        if uiView.previewLayer.session !== session {
            uiView.previewLayer.session = session
        }
    }
}

/// Backing UIView whose layer class is `AVCaptureVideoPreviewLayer`.
/// Using the host view's layer (rather than adding a sublayer) keeps
/// the preview perfectly in sync with autolayout-driven resizing.
final class PreviewContainerView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        // swiftlint:disable:next force_cast
        layer as! AVCaptureVideoPreviewLayer
    }
}
