import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:isolate';
import 'package:flutter/foundation.dart'; // For compute()
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img;

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    _cameras = await availableCameras();
  } on CameraException catch (e) {
    debugPrint('Error initializing cameras: $e');
  }
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? controller;
  bool isStreaming = false;
  String debugStatus = "Initializing...";
  
  // Throttling Control
  bool isProcessingFrame = false; 
  DateTime? lastFrameTime;

  @override
  void initState() {
    super.initState();
    _requestPermissions();
  }

  Future<void> _requestPermissions() async {
    var status = await Permission.camera.request();
    if (status.isGranted) {
      _initCamera();
    } else {
      setState(() => debugStatus = "Permission Denied");
    }
  }

  void _initCamera() {
    controller = CameraController(
      _cameras[0], 
      ResolutionPreset.medium, // 480p (640x480) - Good for AI
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420, // Force consistent format on Android
    );

    controller!.initialize().then((_) {
      if (!mounted) return;
      setState(() => debugStatus = "Camera Ready. Tap Start.");
    }).catchError((Object e) {
      if (e is CameraException) {
        debugPrint('Camera Error: ${e.description}');
      }
    });
  }

  void _toggleStream() {
    if (controller == null || !controller!.value.isInitialized) return;

    if (isStreaming) {
      controller!.stopImageStream();
      setState(() {
        isStreaming = false;
        debugStatus = "Stream Paused";
      });
    } else {
      controller!.startImageStream((CameraImage image) {
        _processFrame(image);
      });
      setState(() => isStreaming = true);
    }
  }

  // --- CORE LOGIC: THROTTLING & PROCESSING ---
 Future<void> _processFrame(CameraImage image) async {
    if (isProcessingFrame) return; 
    final now = DateTime.now();
    if (lastFrameTime != null && 
        now.difference(lastFrameTime!) < const Duration(milliseconds: 1500)) {
      return; 
    }

    isProcessingFrame = true;
    lastFrameTime = now;

    //  START STOPWATCH
    final stopwatch = Stopwatch()..start();

    try {
      final rawData = {
        'width': image.width,
        'height': image.height,
        'format': image.format.group,
        'planes': image.planes.map((plane) => {
          'bytes': plane.bytes,
          'bytesPerRow': plane.bytesPerRow,
          'bytesPerPixel': plane.bytesPerPixel,
        }).toList(),
      };

      final String? base64Result = await compute(convertToBase64Jpeg, rawData);

      //  STOP STOPWATCH
      stopwatch.stop();
      final int processTime = stopwatch.elapsedMilliseconds;

      if (base64Result != null) {
        setState(() {
          // Update UI with Latency Stats
          debugStatus = "Sent Frame!\n"
              "Size: ${(base64Result.length / 1024).toStringAsFixed(1)} KB\n"
              "Latency: ${processTime}ms\n" // <--- NEW METRICfl
              "Base64: ${base64Result.substring(0, 20)}...";
        });
        
        // Log it to terminal so you can graph it later
        print(" Payload Ready: ${(base64Result.length / 1024).toStringAsFixed(1)} KB | Time: ${processTime}ms");
      }
    } catch (e) {
      print("Error processing frame: $e");
    } finally {
      isProcessingFrame = false;
    }
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Aura Throttling Test")),
      body: Column(
        children: [
          Expanded(
            child: controller == null || !controller!.value.isInitialized
                ? const Center(child: CircularProgressIndicator())
                : CameraPreview(controller!),
          ),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            color: Colors.black87,
            child: Column(
              children: [
                Text(
                  debugStatus,
                  style: const TextStyle(color: Colors.white, fontSize: 14, fontFamily: 'monospace'),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _toggleStream,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: isStreaming ? Colors.red : Colors.green,
                  ),
                  child: Text(isStreaming ? "STOP EYES" : "ACTIVATE EYES"),
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// --- ISOLATE FUNCTION (Runs in Background) ---
// This must be a top-level function (outside any class)
Future<String?> convertToBase64Jpeg(Map<String, dynamic> data) async {
  try {
    final width = data['width'] as int;
    final height = data['height'] as int;
    final format = data['format'] as ImageFormatGroup;
    final planes = data['planes'] as List;

    img.Image? image;

    // Convert YUV420 (Android) to RGB
    if (format == ImageFormatGroup.yuv420) {
      // Note: This is a simplified YUV converter for demo speed.
      // For production, use the full 'image' package YUV conversion logic.
      image = img.Image(width: width, height: height);
      final yPlane = planes[0]['bytes'] as Uint8List;
      // Just using Y plane (Greyscale) is faster for testing and often sufficient for AI text reading
      // If you need color, you must merge U and V planes (computationally expensive in Dart)
      for (var y = 0; y < height; y++) {
        for (var x = 0; x < width; x++) {
          final pixel = yPlane[y * width + x];
          image.setPixelRgb(x, y, pixel, pixel, pixel);
        }
      }
    } 
    // Convert BGRA8888 (iOS) to RGB
    else if (format == ImageFormatGroup.bgra8888) {
      final bytes = planes[0]['bytes'] as Uint8List;
      image = img.Image.fromBytes(
        width: width, 
        height: height, 
        bytes: bytes.buffer,
        order: img.ChannelOrder.bgra
      );
    }

    if (image == null) return null;

    // 3. Resize to 640x480 (VGA)
    final resized = img.copyResize(image, width: 640); // Height auto-scales

    // 4. Compress to JPEG (Quality 70)
    final jpegBytes = img.encodeJpg(resized, quality: 70);

    // 5. Convert to Base64
    return base64Encode(jpegBytes);

  } catch (e) {
    print("Isolate Error: $e");
    return null;
  }
}