import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:isolate';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'services/audio_player_service.dart';
import 'screens/login_screen.dart'; // Make sure you created this file in Step 1!

late List<CameraDescription> _cameras;

// 1. MAIN ENTRY POINT (Must be top-level)
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(); // Initialize Firebase
  
  try {
    _cameras = await availableCameras();
  } on CameraException catch (e) {
    debugPrint('Error initializing cameras: $e');
  }
  runApp(const MyApp());
}

// 2. ROOT WIDGET (Handles Auth State)
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData.dark(),
      // Listens to Auth State: If logged in -> Camera, else -> Login
      home: StreamBuilder<User?>(
        stream: FirebaseAuth.instance.authStateChanges(),
        builder: (context, snapshot) {
          if (snapshot.hasData) {
            return const CameraScreen(); 
          }
          return const LoginScreen();
        },
      ),
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
  
  bool isProcessingFrame = false; 
  DateTime? lastFrameTime;

  final PcmAudioService _audioService = PcmAudioService();
  bool _isTestingAudio = false;

  // REPLACE WITH YOUR LAPTOP IP!
  final String _socketUrl = 'ws://192.168.0.61:8080/ws';
  WebSocketChannel? _channel;
  bool _isConnected = false;

  @override
  void initState() {
    super.initState();
    _requestPermissions();
    _initAudio();
    // Connect logic handles the token now
    _connectWebSocket(); 
  }

  // --- NEW: SECURE CONNECTION LOGIC ---
  Future<void> _connectWebSocket() async {
    try {
      // 1. GET FIREBASE TOKEN
      final user = FirebaseAuth.instance.currentUser;
      final token = await user?.getIdToken();

      if (token == null) {
        print("❌ No Auth Token Found. Cannot connect.");
        return;
      }

      // 2. ATTACH TOKEN TO URL
      final secureUrl = "$_socketUrl?token=$token";
      print("🔌 Connecting with Token...");

      _channel = WebSocketChannel.connect(Uri.parse(secureUrl));
      
      // 3. UPDATE UI IMMEDIATELY
      setState(() => _isConnected = true);

      _channel!.stream.listen((message) {
        if (!_isConnected) setState(() => _isConnected = true);

        try {
          final data = jsonDecode(message);
          if (data.containsKey('audio')) {
            final audioBytes = base64Decode(data['audio']);
            // print("🔊 Phone Received Audio: ${audioBytes.length} bytes");
            
            if (_audioService.isInitialized) {
              _audioService.feedAudioChunk(audioBytes);
            }
          }
        } catch (e) {
          print("Error parsing server message: $e");
        }
      }, onError: (error) {
        print("WebSocket Error: $error");
        setState(() => _isConnected = false);
      }, onDone: () {
        print("WebSocket Closed");
        setState(() => _isConnected = false);
      });
    } catch (e) {
      print("Connection Failed: $e");
    }
  }

  Future<void> _initAudio() async {
    await _audioService.initialize();
    await _audioService.start(); 
  }

  Future<void> _requestPermissions() async {
    await [Permission.camera, Permission.microphone].request();
    _initCamera();
  }

  void _initCamera() {
    if (_cameras.isEmpty) return;
    controller = CameraController(
      _cameras[0],
      ResolutionPreset.medium, 
      enableAudio: false, 
      imageFormatGroup: ImageFormatGroup.yuv420, 
    );

    controller!.initialize().then((_) {
      if(!mounted) return;
      setState(() => debugStatus = "Camera Ready. Tap to Start.");
    }).catchError((Object e) {
      if (e is CameraException) {
        debugPrint('Camera Error: ${e.description}');
      }
    });
  }

  void _testAudioOutput() {
    print('Audio Test Disabled (Using Real AI Audio now)');
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

  Future<void> _processFrame(CameraImage image) async {
    if (isProcessingFrame) return; 
    final now = DateTime.now();
    if (lastFrameTime != null && 
        now.difference(lastFrameTime!) < const Duration(milliseconds: 1500)) {
      return; 
    }

    isProcessingFrame = true;
    lastFrameTime = now;

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

      stopwatch.stop();
      final int processTime = stopwatch.elapsedMilliseconds;

      if (base64Result != null) {
        if (_channel != null && _isConnected) {
            _channel!.sink.add(jsonEncode({
                "image": base64Result
            }));
        }

        setState(() {
          debugStatus = "Sent to Brain!\n" 
              "Size: ${(base64Result.length / 1024).toStringAsFixed(1)} KB\n"
              "Latency: ${processTime}ms\n"
              "Connected: $_isConnected"; 
        });
        print("Payload Sent | Time: ${processTime}ms");
      }
    } catch (e) {
      print("Error processing frame: $e");
    } finally {
      isProcessingFrame = false;
    }
  }

  @override
  void dispose() {
    _channel?.sink.close();
    _audioService.dispose();
    controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // UPDATED APP BAR WITH LOGOUT
      appBar: AppBar(
        title: const Text("Aura Vision"),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () async {
              await FirebaseAuth.instance.signOut();
            },
          )
        ],
      ),
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
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    ElevatedButton(
                      onPressed: _toggleStream,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: isStreaming? Colors.red : Colors.green,
                      ),
                      child: Text(isStreaming ? "STOP EYES" : "ACTIVATE EYES"),
                    ),
                    ElevatedButton(
                      onPressed: _testAudioOutput, 
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _isConnected ? Colors.blue : Colors.grey,
                      ),
                      child: Text(_isConnected ? "CONNECTED" : "OFFLINE"),
                    ),
                  ],
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}

Future<String?> convertToBase64Jpeg(Map<String, dynamic> data) async {
  try {
    final width = data['width'] as int;
    final height = data['height'] as int;
    final format = data['format'] as ImageFormatGroup;
    final planes = data['planes'] as List;

    img.Image? image;

    if (format == ImageFormatGroup.yuv420) {
      image = img.Image(width: width, height: height);
      final yPlane = planes[0]['bytes'] as Uint8List;
      for (var y = 0; y < height; y++) {
        for (var x = 0; x < width; x++) {
          final pixel = yPlane[y * width + x];
          image.setPixelRgb(x, y, pixel, pixel, pixel);
        }
      }
    } else if (format == ImageFormatGroup.bgra8888) {
      final bytes = planes[0]['bytes'] as Uint8List;
      image = img.Image.fromBytes(
        width: width, 
        height: height, 
        bytes: bytes.buffer,
        order: img.ChannelOrder.bgra
      );
    }

    if (image == null) return null;

    final resized = img.copyResize(image, width: 640); 
    final jpegBytes = img.encodeJpg(resized, quality: 60);
    return base64Encode(jpegBytes);

  } catch (e) {
    print("Isolate Error: $e");
    return null;
  }
}