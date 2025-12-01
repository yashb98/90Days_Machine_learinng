import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart'; // For compute()
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_tts/flutter_tts.dart'; 
import 'package:firebase_analytics/firebase_analytics.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? controller;
  bool isStreaming = false;
  String debugStatus = "Initializing...";
  String aiStatus = "IDLE"; 
  
  bool isProcessingFrame = false; 
  DateTime? lastFrameTime;

  late FlutterTts flutterTts;
  
  // REPLACE WITH YOUR CLOUD RUN URL (wss://...)
  final String _socketUrl = 'wss://aura-backend-service-963226949438.europe-west2.run.app/ws';
  WebSocketChannel? _channel;
  bool _isConnected = false;

  @override
  void initState() {
    super.initState();
    _initTts(); 
    _requestPermissions();
    _connectWebSocket(); 
  }

  void _initTts() async {
    flutterTts = FlutterTts();
    
    await flutterTts.setLanguage("en-US");
    await flutterTts.setPitch(1.0);
    await flutterTts.setSpeechRate(0.5);
    await flutterTts.awaitSpeakCompletion(true); // Wait for speech to finish

    // Android specific audio focus settings
    // This ensures music pauses when Aura speaks
    await flutterTts.setIosAudioCategory(IosTextToSpeechAudioCategory.playback,
        [
          IosTextToSpeechAudioCategoryOptions.defaultToSpeaker,
          IosTextToSpeechAudioCategoryOptions.duckOthers
        ],
    );

    flutterTts.setStartHandler(() {
      print("TTS: Started playing");
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
        print("❌ No Auth Token Found.");
        return;
      }

      final secureUri = Uri.parse(_socketUrl).replace(queryParameters: {'token': token});
      print("🔌 Connecting to: $secureUri");

      _channel = WebSocketChannel.connect(secureUri);
      if(mounted) setState(() => _isConnected = true);

      _channel!.stream.listen((message) {
        if (!_isConnected && mounted) setState(() => _isConnected = true);

        // Debug Log
        print("📩 RECEIVED: $message");

        try {
          final data = jsonDecode(message);
          
          // 1. HANDLE SPEAK
          if (data['cmd'] == 'speak') {
             String text = data['text'];
             print("🗣️ AI COMMAND: Speak -> $text");
             
             // Update UI to show what it's saying
             if(mounted) {
               setState(() {
                 aiStatus = "SPEAKING";
                //  debugStatus = "AI: $text";
               });
             }

             // Execute Speech
             _speak(text);
          }
          
          // 2. HANDLE STATUS UPDATES
          else if (data['cmd'] == 'status') {
            String newState = data['state'].toString().toUpperCase();
            print("🔄 STATUS UPDATE: $newState");
            
            if(mounted) {
              setState(() {
                aiStatus = newState;
              });
            }
          }
          
          // 3. INTERRUPT
          else if (data['cmd'] == 'interrupt') {
             print("🛑 INTERRUPT COMMAND");
             flutterTts.stop();
             if(mounted) setState(() => aiStatus = "INTERRUPTED");
          }

        } catch (e) {
          print("Error parsing message: $e");
        }
      }, onError: (error) {
        print("WebSocket Error: $error");
        if(mounted) setState(() => _isConnected = false);
      }, onDone: () {
        print("WebSocket Closed");
        if(mounted) setState(() => _isConnected = false);
      });
    } catch (e) {
      print("Connection Failed: $e");
    }
  }
  
  Future<void> _speak(String text) async {
    await flutterTts.stop(); // Stop previous speech
    if (text.isNotEmpty) {
      await flutterTts.speak(text);
    }
  }

  Future<void> _requestPermissions() async {
    await [Permission.camera, Permission.microphone].request();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;
    
    controller = CameraController(
      cameras[0],
      ResolutionPreset.medium, 
      enableAudio: false, 
      imageFormatGroup: ImageFormatGroup.yuv420, 
    );

    await controller!.initialize();
    if (!mounted) return;
    setState(() => debugStatus = "Camera Ready. Tap Activate.");
  }

  void _toggleStream() {
    if (controller == null || !controller!.value.isInitialized) return;

    if (isStreaming) {
      controller!.stopImageStream();
      setState(() {
        isStreaming = false;
        debugStatus = "Stream Paused";
        aiStatus = "IDLE";
      });
    } else {
      controller!.startImageStream((CameraImage image) {
        _processFrame(image);
      });
      setState(() {
        isStreaming = true;
        aiStatus = "WATCHING";
      });
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
        _channel!.sink.add(jsonEncode({
            "image": base64Result
        }));
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
    flutterTts.stop();
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
                            color: _getStatusColor(aiStatus),
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
                        minimumSize: const Size(150, 50),
                      ),
                      child: Text(
                        isStreaming ? "STOP EYES" : "ACTIVATE EYES",
                        style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                      ),
                    ),
                    Icon(
                      _isConnected ? Icons.cloud_done : Icons.cloud_off,
                      color: _isConnected ? Colors.blue : Colors.grey,
                      size: 30,
                    )
                  ],
                )
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color _getStatusColor(String status) {
    switch (status) {
      case "THINKING": return Colors.yellow;
      case "SPEAKING": return Colors.blue;
      case "READY": return Colors.green;
      case "IDLE": return Colors.grey;
      default: return Colors.white;
    }
  }
}

// --- ISOLATE FUNCTION (Background Thread) ---
Future<String?> convertToBase64Jpeg(Map<String, dynamic> data) async {
  try {
    final width = data['width'] as int;
    final height = data['height'] as int;
    final format = data['format'] as ImageFormatGroup;
    final planes = data['planes'] as List;

    img.Image? image;

    if (format == ImageFormatGroup.yuv420) {
      // Optimized Grayscale Conversion (Y-Plane only)
      image = img.Image(width: width, height: height, numChannels: 1); 
      
      final yPlane = planes[0]['bytes'] as Uint8List;
      final int bytesPerRow = planes[0]['bytesPerRow'] as int; 

      for (var y = 0; y < height; y++) {
        for (var x = 0; x < width; x++) {
          final int yIndex = y * bytesPerRow + x;
          if (yIndex < yPlane.length) {
             final pixel = yPlane[yIndex];
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

    // Resize to VGA (640x480) for speed
    final resized = img.copyResize(image, width: 640); 
    final jpegBytes = img.encodeJpg(resized, quality: 70); 
    return base64Encode(jpegBytes);

  } catch (e) {
    print("Isolate Error: $e");
    return null;
  }
}