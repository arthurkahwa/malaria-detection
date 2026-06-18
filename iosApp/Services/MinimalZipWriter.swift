import Foundation

/// Hand-rolled ZIP writer that emits a minimal, **stored** (uncompressed)
/// PKZIP archive containing the small text payloads we need for the
/// Phase 13 export bundle (spec §14): `export.json` + `README.txt`.
///
/// Why a custom writer rather than `Foundation.FileManager.zipItem(at:to:)`
/// or `Compression.framework`:
///
///  - `FileManager.zipItem(...)` is macOS-only; on iOS the corresponding
///    `NSFileCoordinator(filePresenter:)` ZIP API isn't part of the public
///    surface.
///  - `Compression.framework` provides DEFLATE primitives but not the PKZIP
///    container layout.
///  - Third-party libraries (`ZIPFoundation`, `Zip`) would add an external
///    dependency for ~80 bytes of structural overhead per entry.
///
/// The structure produced here is a strict subset of the ZIP spec
/// (APPNOTE.TXT 6.3.10):
///
///   * one local file header + raw bytes per entry (no compression, so
///     `comp size == uncomp size`),
///   * one central directory entry per file,
///   * one end-of-central-directory record at the tail.
///
/// CRC-32 is computed over the entry bytes using the canonical IEEE 802.3
/// polynomial (`0xEDB88320`, reflected). The "Files" app, `unzip(1)`, and
/// every consumer tested can read the resulting archive.
enum MinimalZipWriter {

    /// One archive entry: filename + raw bytes.
    typealias Entry = (name: String, data: Data)

    /// Errors emitted by the writer.
    enum WriteError: Error, LocalizedError {
        case ioFailure(String)

        var errorDescription: String? {
            switch self {
            case .ioFailure(let reason): return "ZIP write failed: \(reason)"
            }
        }
    }

    /// Write a stored-mode ZIP archive at [destination] containing each of
    /// [entries] as an uncompressed file.
    static func writeArchive(entries: [Entry], to destination: URL) throws {
        var output = Data()
        var centralDirectory = Data()
        var entryCount: UInt16 = 0
        var centralDirectoryOffset: UInt32 = 0

        for entry in entries {
            let nameBytes = Data(entry.name.utf8)
            let payload = entry.data
            let crc = crc32(payload)
            let size = UInt32(payload.count)

            let localHeaderOffset = UInt32(output.count)

            // Local file header --------------------------------------------
            var local = Data()
            local.append(uint32: 0x04034b50)            // signature
            local.append(uint16: 20)                    // version needed: 2.0
            local.append(uint16: 0)                     // GP flag bits
            local.append(uint16: 0)                     // method: stored
            local.append(uint16: 0)                     // last mod time (omitted)
            local.append(uint16: 0)                     // last mod date (omitted)
            local.append(uint32: crc)                   // CRC-32
            local.append(uint32: size)                  // compressed size
            local.append(uint32: size)                  // uncompressed size
            local.append(uint16: UInt16(nameBytes.count))
            local.append(uint16: 0)                     // extra field length
            local.append(nameBytes)
            output.append(local)
            output.append(payload)

            // Central directory entry --------------------------------------
            var central = Data()
            central.append(uint32: 0x02014b50)          // signature
            central.append(uint16: 20)                  // version made by
            central.append(uint16: 20)                  // version needed
            central.append(uint16: 0)                   // GP flag
            central.append(uint16: 0)                   // method
            central.append(uint16: 0)                   // last mod time
            central.append(uint16: 0)                   // last mod date
            central.append(uint32: crc)                 // CRC-32
            central.append(uint32: size)                // compressed size
            central.append(uint32: size)                // uncompressed size
            central.append(uint16: UInt16(nameBytes.count))
            central.append(uint16: 0)                   // extra field length
            central.append(uint16: 0)                   // comment length
            central.append(uint16: 0)                   // disk number start
            central.append(uint16: 0)                   // internal attrs
            central.append(uint32: 0)                   // external attrs
            central.append(uint32: localHeaderOffset)
            central.append(nameBytes)
            centralDirectory.append(central)

            entryCount += 1
        }

        centralDirectoryOffset = UInt32(output.count)
        output.append(centralDirectory)

        // End of central directory record ----------------------------------
        var eocd = Data()
        eocd.append(uint32: 0x06054b50)                 // signature
        eocd.append(uint16: 0)                          // disk number
        eocd.append(uint16: 0)                          // disk where CD starts
        eocd.append(uint16: entryCount)                 // entries on this disk
        eocd.append(uint16: entryCount)                 // total entries
        eocd.append(uint32: UInt32(centralDirectory.count))
        eocd.append(uint32: centralDirectoryOffset)
        eocd.append(uint16: 0)                          // comment length
        output.append(eocd)

        do {
            try output.write(to: destination, options: [.atomic])
        } catch {
            throw WriteError.ioFailure(error.localizedDescription)
        }
    }

    // MARK: - CRC-32 (IEEE 802.3, reflected)

    private static let crc32Table: [UInt32] = {
        var table = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            var c = UInt32(i)
            for _ in 0..<8 {
                c = (c & 1 == 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1)
            }
            table[i] = c
        }
        return table
    }()

    static func crc32(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFFFFFF
        for byte in data {
            let index = Int((c ^ UInt32(byte)) & 0xFF)
            c = (c >> 8) ^ crc32Table[index]
        }
        return c ^ 0xFFFFFFFF
    }
}

// MARK: - Little-endian Data append helpers

private extension Data {
    mutating func append(uint16 value: UInt16) {
        var v = value.littleEndian
        Swift.withUnsafeBytes(of: &v) { append(contentsOf: $0) }
    }
    mutating func append(uint32 value: UInt32) {
        var v = value.littleEndian
        Swift.withUnsafeBytes(of: &v) { append(contentsOf: $0) }
    }
}
