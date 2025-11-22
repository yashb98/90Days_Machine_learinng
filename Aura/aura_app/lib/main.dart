import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';

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
  int frameCount = 0;
  String debugStatus = "Initializing..."; // New variable for on-screen logs

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
    // Using first camera (usually back)
    controller = CameraController(_cameras[0], ResolutionPreset.medium, enableAudio: false);

    controller!.initialize().then((_) {
      if (!mounted) return;
      setState(() {
        debugStatus = "Camera Ready. Tap Start.";
      });
    }).catchError((Object e) {
      if (e is CameraException) {
        debugPrint('Camera Error: ${e.description}');
      }
    });
  }

  // NEW FUNCTION: Controls the Start/Stop logic
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
        frameCount++;
        
        // Show log on screen every 10 frames
        if (frameCount % 10 == 0) {
          setState(() {
            debugStatus = "Frames: $frameCount\nRes: ${image.width}x${image.height}\nFormat: ${image.format.group}";
          });
          
          // Also print to terminal for verification
          print(" FRAME CAPTURED | Bytes: ${image.planes[0].bytes.length}");
        }
      });
      setState(() {
        isStreaming = true;
      });
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
      appBar: AppBar(title: const Text("Aura Vision Test")),
      // NEW UI STRUCTURE: Column to hold Camera + Controls
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
                  debugStatus, // Displaying logs here
                  style: const TextStyle(color: Colors.white, fontSize: 16, fontFamily: 'monospace'),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _toggleStream, // Hooked up to the toggle function
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