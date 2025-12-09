import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

Future<String?> convertToBase64Jpeg(Map<String, dynamic> data) async {
  try {
    final width = data['width'] as int;
    final height = data['height'] as int;
    final format = data['format'] as ImageFormatGroup;
    final planes = data['planes'] as List;

    if (format == ImageFormatGroup.yuv420) {
      final Uint8List yBytes = planes[0]['bytes'];
      final int yRowStride = planes[0]['bytesPerRow'];

      final img.Image image = img.Image.fromBytes(
        width: width,
        height: height,
        bytes: yBytes.buffer,
        rowStride: yRowStride,
        numChannels: 1 
      );

      final img.Image rotated = img.copyRotate(image, angle: 90);
      final img.Image resized = img.copyResize(rotated, width: 320);
      final jpegBytes = img.encodeJpg(resized, quality: 60);
      
      return base64Encode(jpegBytes);
    } 
    return null; 
  } catch (e) {
    print("❌ Image Convert Error: $e");
    return null;
  }
}