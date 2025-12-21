import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

Future<String?> convertToBase64Jpeg(Map<String, dynamic> data) async {
  try {
    final width = data['width'];
    final height = data['height'];
    final planes = data['planes'];

    // YUV420 to RGB Conversion (Keeps Colors!)
    // simplified for performance: only strictly accurate for standard Android Camera2 API
    final yPlane = planes[0]['bytes'] as Uint8List;
    final uPlane = planes[1]['bytes'] as Uint8List;
    final vPlane = planes[2]['bytes'] as Uint8List;

    final int yRowStride = planes[0]['bytesPerRow'];
    final int uvRowStride = planes[1]['bytesPerRow'];
    final int uvPixelStride = planes[1]['bytesPerPixel'];

    img.Image image = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int yIndex = y * yRowStride + x;
        final int uvIndex = (y ~/ 2) * uvRowStride + (x ~/ 2) * uvPixelStride;

        // YUV Indices
        final int yp = yPlane[yIndex];
        final int up = uPlane[uvIndex];
        final int vp = vPlane[uvIndex];

        // Standard YUV conversion formula
        int r = (yp + (1.370705 * (vp - 128))).round().clamp(0, 255);
        int g = (yp - (0.337633 * (up - 128)) - (0.698001 * (vp - 128))).round().clamp(0, 255);
        int b = (yp + (1.732446 * (up - 128))).round().clamp(0, 255);

        image.setPixelRgb(x, y, r, g, b);
      }
    }

    // Rotate (Portrait) & Resize
    img.Image oriented = img.copyRotate(image, angle: 90);
    img.Image resized = img.copyResize(oriented, width: 640); // 640px is plenty for AI

    return base64Encode(img.encodeJpg(resized, quality: 70));
  } catch (e) {
    print("Convert Error: $e");
    return null;
  }
}