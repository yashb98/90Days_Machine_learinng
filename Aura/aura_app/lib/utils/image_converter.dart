import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

Future<String?> convertToBase64Jpeg(Map<String, dynamic> data) async {
  try {
    final width = data['width'] as int;
    final height = data['height'] as int;
    final format = data['format'] as String;
    final planes = data['planes'] as List;

    img.Image? finalImage;

    // --- 1. ANDROID (YUV420) -> FORCE RGB GRAYSCALE ---
    if (format == 'yuv420') {
      final Uint8List yBytes = planes[0]['bytes'];
      final int yRowStride = planes[0]['bytesPerRow'];
      
      // FIX: Create a 3-Channel RGB image (Not 1-channel)
      // This prevents the "Red Tint" interpretation.
      finalImage = img.Image(width: width, height: height, numChannels: 3);

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int uvIndex = y * yRowStride + x;
          
          if (uvIndex < yBytes.length) {
            final int brightness = yBytes[uvIndex];
            
            // FIX: Set R, G, and B to the SAME value.
            // When R=G=B, the pixel is perfectly gray.
            finalImage.setPixelRgb(x, y, brightness, brightness, brightness); 
          }
        }
      }
    } 
    // --- 2. iOS (BGRA8888) ---
    else if (format == 'bgra8888') {
      final Uint8List bytes = planes[0]['bytes'];
      final int bytesPerRow = planes[0]['bytesPerRow'];

      finalImage = img.Image.fromBytes(
        width: width,
        height: height,
        bytes: bytes.buffer,
        rowStride: bytesPerRow,
        numChannels: 4,
        order: img.ChannelOrder.bgra,
      );
    }

    if (finalImage == null) return null;

    // --- 3. ROTATION & RESIZING ---
    // Rotate 90 degrees (Portrait Mode)
    img.Image oriented = img.copyRotate(finalImage, angle: 90);

    // Resize to 640px (Good balance for AI)
    img.Image resized = img.copyResize(
      oriented, 
      width: 640, 
      interpolation: img.Interpolation.nearest
    );

    // --- 4. ENCODE TO JPEG ---
    final jpegBytes = img.encodeJpg(resized, quality: 70);

    return base64Encode(jpegBytes);

  } catch (e) {
    print("Image Convert Error: $e");
    return null;
  }
}