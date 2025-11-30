import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_tts/flutter_tts.dart'; 
import 'screens/login_screen.dart';
import 'package:firebase_analytics/firebase_analytics.dart';



late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  
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
    return MaterialApp(
      theme: ThemeData.dark(),
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
  String aiStatus = "IDLE"; // To show "Thinking" vs "Ready"
  
  bool isProcessingFrame = false; 
  DateTime? lastFrameTime;

  // --- CHANGED: Use FlutterTts instead of PcmAudioService ---
  late FlutterTts flutterTts;
  
  // REPLACE WITH YOUR LAPTOP IP!
  final String _socketUrl = 'ws://aura-backend-service-963226949438.europe-west2.run.app/ws';
  WebSocketChannel? _channel;
  bool _isConnected = false;

  @override
  void initState() {
    super.initState();
    _initTts(); // Initialize TTS
    _requestPermissions();
    _connectWebSocket(); 
  }

  // --- NEW: TTS INITIALIZATION ---
  void _initTts() {
    flutterTts = FlutterTts();
    
    // Configure voice settings
    flutterTts.setLanguage("en-US");
    flutterTts.setPitch(1.0);
    flutterTts.setSpeechRate(0.5); // Normal speed
    
    // Handler to check if it's working
    flutterTts.setStartHandler(() {
      print("TTS Started playing");
    });
    
    flutterTts.setCompletionHandler(() {
      print("TTS Finished");
    });
    
    flutterTts.setErrorHandler((msg) {
      print("TTS Error: $msg");
    });
  }

  Future<void> _connectWebSocket() async {
    try {
      final user = FirebaseAuth.instance.currentUser;
      final token = await user?.getIdToken();

      if (token == null) {
        print("No Auth Token Found. Cannot connect.");
        return;
      }

      final secureUrl = "$_socketUrl?token=$token";
      print("Connecting with Token...");

      _channel = WebSocketChannel.connect(Uri.parse(secureUrl));
      setState(() => _isConnected = true);

      _channel!.stream.listen((message) {
        // --- THIS IS THE CRITICAL FIX ---
        // We now listen for JSON commands, not raw audio bytes
        try {
          final data = jsonDecode(message);
          
          // 1. HANDLE SPEAK COMMAND
          if (data['cmd'] == 'speak') {
             String text = data['text'];
             print("AI Says: $text");
             flutterTts.speak(text);
          }
          
          // 2. HANDLE INTERRUPT (Barge-In)
          else if (data['cmd'] == 'interrupt') {
            print("Interrupted!");
            flutterTts.stop();
          }
          
          // 3. HANDLE STATUS UPDATES
          else if (data['cmd'] == 'status') {
            setState(() {
              aiStatus = data['state'].toString().toUpperCase();
            });
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
      setState(() => debugStatus = "Camera Ready. Tap Activate.");
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

  Future<void> _processFrame(CameraImage image) async {
    if (isProcessingFrame) return; 
    final now = DateTime.now();
    // Send frame every 1.5 seconds (Balance between lag and realtime)
    if (lastFrameTime != null && 
        now.difference(lastFrameTime!) < const Duration(milliseconds: 1500)) {
      return; 
    }

    isProcessingFrame = true;
    lastFrameTime = now;

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

      if (base64Result != null && _channel != null && _isConnected) {
        // Send image to backend
        _channel!.sink.add(jsonEncode({
            "image": base64Result
        }));
        
        // Update UI
        setState(() {
          debugStatus = "Status: $aiStatus\nConnected: $_isConnected"; 
        });
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
    flutterTts.stop(); // Stop speaking on exit
    controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
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
                : Stack(
                    children: [
                      CameraPreview(controller!),
                      // OVERLAY FOR AI STATUS
                      Positioned(
                        top: 20,
                        right: 20,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                          decoration: BoxDecoration(
                            color: aiStatus == "THINKING" ? Colors.yellow : Colors.green,
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            aiStatus,
                            style: const TextStyle(
                              color: Colors.black, 
                              fontWeight: FontWeight.bold
                            ),
                          ),
                        ),
                      )
                    ],
                  ),
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
                      onPressed: () {
                         // Manual Audio Test
                         flutterTts.speak("System Online. Audio test complete.");
                      }, 
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                      ),
                      child: const Text("TEST AUDIO"),
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
      // Use Grayscale image container for Y-plane data
      image = img.Image(width: width, height: height, numChannels: 1); 
      
      final yPlane = planes[0]['bytes'] as Uint8List;
      final int bytesPerRow = planes[0]['bytesPerRow'] as int; // <--- VITAL

      for (var y = 0; y < height; y++) {
        for (var x = 0; x < width; x++) {
          // Use bytesPerRow to skip padding bytes
          final int uvIndex = y * bytesPerRow + x;
          
          // Safety check to avoid crashing on edge pixels
          if (uvIndex < yPlane.length) {
             final pixel = yPlane[uvIndex];
             // Set grayscale pixel (r=g=b=pixel)
             image.setPixelRgb(x, y, pixel, pixel, pixel);
          }
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

    // Resize to reduce latency (320px is enough for AI)
    final resized = img.copyResize(image, width: 640); 
    final jpegBytes = img.encodeJpg(resized, quality: 120);
    return base64Encode(jpegBytes);

  } catch (e) {
    print("Isolate Error: $e");
    return null;
  }
}
