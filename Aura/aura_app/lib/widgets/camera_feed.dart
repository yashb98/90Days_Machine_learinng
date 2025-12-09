import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class CameraFeed extends StatelessWidget {
  final CameraController? controller;

  const CameraFeed({super.key, required this.controller});

  @override
  Widget build(BuildContext context) {
    if (controller != null && controller!.value.isInitialized) {
      return Center(
        child: CameraPreview(controller!),
      );
    } else {
      return const Center(
        child: CircularProgressIndicator(color: Colors.cyanAccent)
      );
    }
  }
}